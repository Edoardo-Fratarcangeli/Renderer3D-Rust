//! NDJSON (newline-delimited JSON) [`BatchDecoder`].
//!
//! One JSON value per line, each decoded into a single-row [`RowBatch`] via
//! [`parse_json_row`]. The decoder keeps its own byte buffer so a row split
//! across multiple `read` calls is reassembled correctly.

use std::io::Read;

use super::{parse_json_row, BatchDecoder, RowBatch};
use crate::dataset::{DatasetError, Result};

/// Read chunk size for pulling bytes off the transport.
const CHUNK: usize = 16 * 1024;

pub struct NdjsonDecoder<R: Read> {
    reader: R,
    buf: Vec<u8>,
    /// Set once the underlying stream reports EOF.
    eof: bool,
}

impl<R: Read> NdjsonDecoder<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            buf: Vec::with_capacity(CHUNK),
            eof: false,
        }
    }

    /// Pop the next complete line (without the trailing `\n`) from the buffer.
    fn take_line(&mut self) -> Option<Vec<u8>> {
        let pos = self.buf.iter().position(|&b| b == b'\n')?;
        let mut line: Vec<u8> = self.buf.drain(0..=pos).collect();
        line.pop(); // drop '\n'
        if line.last() == Some(&b'\r') {
            line.pop(); // tolerate CRLF
        }
        Some(line)
    }

    fn parse_line(line: &[u8]) -> Result<Option<RowBatch>> {
        if line.iter().all(|b| b.is_ascii_whitespace()) {
            return Ok(None); // skip blank lines
        }
        let value: serde_json::Value = serde_json::from_slice(line)
            .map_err(|e| DatasetError::Format(format!("invalid NDJSON row: {e}")))?;
        parse_json_row(&value).map(Some)
    }
}

impl<R: Read + Send> BatchDecoder for NdjsonDecoder<R> {
    fn next_batch(&mut self) -> Result<Option<RowBatch>> {
        loop {
            // Emit any complete line already buffered (skipping blanks).
            while let Some(line) = self.take_line() {
                if let Some(batch) = Self::parse_line(&line)? {
                    return Ok(Some(batch));
                }
            }
            if self.eof {
                // Flush a trailing line that had no terminating newline.
                if !self.buf.is_empty() {
                    let rest = std::mem::take(&mut self.buf);
                    return Self::parse_line(&rest);
                }
                return Ok(None);
            }

            // Pull more bytes from the transport.
            let mut chunk = [0u8; CHUNK];
            match self.reader.read(&mut chunk) {
                Ok(0) => self.eof = true,
                Ok(n) => self.buf.extend_from_slice(&chunk[..n]),
                Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(DatasetError::Io(e)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// A reader that hands out bytes in fixed-size pieces, to exercise the
    /// cross-read line reassembly.
    struct Choppy {
        data: Vec<u8>,
        pos: usize,
        step: usize,
    }
    impl Read for Choppy {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            if self.pos >= self.data.len() {
                return Ok(0);
            }
            let end = (self.pos + self.step).min(self.data.len());
            let n = (end - self.pos).min(out.len());
            out[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
            self.pos += n;
            Ok(n)
        }
    }

    fn collect<R: Read + Send>(mut d: NdjsonDecoder<R>) -> Vec<RowBatch> {
        let mut out = Vec::new();
        while let Some(b) = d.next_batch().unwrap() {
            out.push(b);
        }
        out
    }

    #[test]
    fn decodes_multiple_lines() {
        let text = "{\"x\":[1,2],\"label\":\"a\"}\n[3,4]\n";
        let batches = collect(NdjsonDecoder::new(Cursor::new(text)));
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].rows, vec![1.0, 2.0]);
        assert_eq!(batches[0].labels, vec!["a".to_string()]);
        assert_eq!(batches[1].rows, vec![3.0, 4.0]);
        assert!(batches[1].labels.is_empty());
    }

    #[test]
    fn reassembles_rows_split_across_reads() {
        let text = "{\"x\":[1,2,3]}\n{\"x\":[4,5,6]}\n";
        // One byte at a time: every line spans many reads.
        let choppy = Choppy {
            data: text.as_bytes().to_vec(),
            pos: 0,
            step: 1,
        };
        let batches = collect(NdjsonDecoder::new(choppy));
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[1].rows, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn flushes_trailing_line_without_newline() {
        let text = "[1,2]\n[3,4]"; // no final '\n'
        let batches = collect(NdjsonDecoder::new(Cursor::new(text)));
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[1].rows, vec![3.0, 4.0]);
    }

    #[test]
    fn skips_blank_lines_and_tolerates_crlf() {
        let text = "[1]\r\n\r\n[2]\r\n";
        let batches = collect(NdjsonDecoder::new(Cursor::new(text)));
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].rows, vec![1.0]);
        assert_eq!(batches[1].rows, vec![2.0]);
    }

    #[test]
    fn surfaces_malformed_json() {
        let mut d = NdjsonDecoder::new(Cursor::new("{not json}\n"));
        assert!(d.next_batch().is_err());
    }
}
