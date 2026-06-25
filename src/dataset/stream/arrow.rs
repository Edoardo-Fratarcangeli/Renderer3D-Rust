//! Apache Arrow IPC stream [`BatchDecoder`] (feature `arrow-stream`).
//!
//! Wraps [`arrow::ipc::reader::StreamReader`]. Numeric columns become feature
//! columns (cast to `Float32`); a single text/categorical column named
//! `label`/`labels`/`target`/`class`/`y` (case-insensitive) becomes the label.
//! This mirrors the tabular label conventions used by the file loaders.

use std::io::Read;

use arrow::array::{Array, Float32Array, StringArray};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use arrow::ipc::reader::StreamReader;
use arrow::record_batch::RecordBatch;

use super::{BatchDecoder, RowBatch};
use crate::dataset::{DatasetError, Result};

pub struct ArrowIpcDecoder<R: Read> {
    reader: StreamReader<R>,
}

impl<R: Read> ArrowIpcDecoder<R> {
    /// Reads the stream schema header (blocking) and prepares the reader.
    pub fn new(reader: R) -> Result<Self> {
        let reader = StreamReader::try_new(reader, None)
            .map_err(|e| DatasetError::Format(format!("Arrow stream header: {e}")))?;
        Ok(Self { reader })
    }
}

fn is_label_name(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "label" | "labels" | "target" | "class" | "y"
    )
}

/// Convert one Arrow record batch into a row-major [`RowBatch`].
fn convert(batch: &RecordBatch) -> Result<RowBatch> {
    let schema = batch.schema();
    let n_rows = batch.num_rows();

    let mut feature_cols: Vec<Float32Array> = Vec::new();
    let mut label_col: Option<StringArray> = None;
    for (i, field) in schema.fields().iter().enumerate() {
        let col = batch.column(i);
        if label_col.is_none() && is_label_name(field.name()) {
            let utf8 = cast(col, &DataType::Utf8)
                .map_err(|e| DatasetError::Format(format!("label column cast: {e}")))?;
            label_col = utf8.as_any().downcast_ref::<StringArray>().cloned();
            continue;
        }
        let f32col = cast(col, &DataType::Float32).map_err(|e| {
            DatasetError::Format(format!("column '{}' is not numeric: {e}", field.name()))
        })?;
        let arr = f32col
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| DatasetError::Format("Float32 downcast failed".into()))?
            .clone();
        feature_cols.push(arr);
    }

    let n_cols = feature_cols.len();
    if n_cols == 0 {
        return Err(DatasetError::Format(
            "Arrow batch has no numeric feature columns".into(),
        ));
    }

    // Row-major flatten (null feature values become 0.0).
    let mut rows = Vec::with_capacity(n_rows * n_cols);
    for r in 0..n_rows {
        for col in &feature_cols {
            rows.push(if col.is_null(r) { 0.0 } else { col.value(r) });
        }
    }
    let labels = match &label_col {
        Some(arr) => (0..n_rows)
            .map(|r| {
                if arr.is_null(r) {
                    String::new()
                } else {
                    arr.value(r).to_string()
                }
            })
            .collect(),
        None => Vec::new(),
    };

    Ok(RowBatch {
        n_cols,
        rows,
        labels,
    })
}

impl<R: Read + Send> BatchDecoder for ArrowIpcDecoder<R> {
    fn next_batch(&mut self) -> Result<Option<RowBatch>> {
        match self.reader.next() {
            None => Ok(None),
            Some(Ok(batch)) => convert(&batch).map(Some),
            Some(Err(e)) => Err(DatasetError::Format(format!("Arrow stream: {e}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{Field, Schema};
    use arrow::ipc::writer::StreamWriter;
    use std::sync::Arc;

    fn sample_ipc_bytes() -> Vec<u8> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("f0", DataType::Float64, false),
            Field::new("f1", DataType::Int32, false),
            Field::new("label", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Int32Array::from(vec![10, 20, 30])),
                Arc::new(StringArray::from(vec!["a", "b", "a"])),
            ],
        )
        .unwrap();
        let mut buf = Vec::new();
        {
            let mut w = StreamWriter::try_new(&mut buf, &schema).unwrap();
            w.write(&batch).unwrap();
            w.finish().unwrap();
        }
        buf
    }

    #[test]
    fn decodes_numeric_columns_and_label() {
        let bytes = sample_ipc_bytes();
        let mut d = ArrowIpcDecoder::new(std::io::Cursor::new(bytes)).unwrap();
        let batch = d.next_batch().unwrap().unwrap();
        assert_eq!(batch.n_cols, 2);
        assert_eq!(batch.n_rows(), 3);
        assert_eq!(batch.rows, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
        assert_eq!(
            batch.labels,
            vec!["a".to_string(), "b".to_string(), "a".to_string()]
        );
        assert!(d.next_batch().unwrap().is_none());
    }
}
