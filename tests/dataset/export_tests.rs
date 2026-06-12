use std::collections::HashSet;

use rendering_3d::dataset::export::export_csv;
use rendering_3d::dataset::index::{apply_filter, DatasetIndex, FilterSpec, SearchQuery};
use rendering_3d::dataset::loader::{load, LoadOptions};

use crate::helpers;

#[test]
fn export_writes_filtered_subset_consistent_with_filter() {
    let dir = tempfile::tempdir().unwrap();
    let src = helpers::write_sample_csv(dir.path());
    let ds = load(&src, &LoadOptions::default()).unwrap();
    let idx = DatasetIndex::build(&ds.labels, ds.label_names.len());

    // Filter: only class_1 -> rows 2 and 3.
    let spec = FilterSpec {
        enabled_labels: HashSet::from([1]),
        query: SearchQuery::All,
    };
    let rows = apply_filter(&ds, &idx, &spec);
    assert_eq!(rows, vec![2, 3]);

    let out = dir.path().join("subset.csv");
    let written = export_csv(&ds, &rows, &out).unwrap();
    assert_eq!(written, 2);

    // The exported file must re-import to exactly the filtered subset.
    let exported = load(&out, &LoadOptions::default()).unwrap();
    assert_eq!(exported.n_rows(), 2);
    assert_eq!(exported.n_cols(), 3);
    assert_eq!(exported.metadata.column_names, vec!["a", "b", "c"]);
    assert_eq!(exported.label_names, vec!["class_1"]);
    let mut row = Vec::new();
    exported.row(0, &mut row);
    let mut orig = Vec::new();
    ds.row(2, &mut orig);
    assert_eq!(row, orig);
}

#[test]
fn export_of_empty_selection_writes_header_only() {
    let dir = tempfile::tempdir().unwrap();
    let src = helpers::write_sample_csv(dir.path());
    let ds = load(&src, &LoadOptions::default()).unwrap();

    let out = dir.path().join("empty.csv");
    assert_eq!(export_csv(&ds, &[], &out).unwrap(), 0);
    let text = std::fs::read_to_string(&out).unwrap();
    assert_eq!(text, "a,b,c,label\n");
}

#[test]
fn export_rejects_out_of_range_rows() {
    let dir = tempfile::tempdir().unwrap();
    let src = helpers::write_sample_csv(dir.path());
    let ds = load(&src, &LoadOptions::default()).unwrap();
    let out = dir.path().join("oob.csv");
    assert!(export_csv(&ds, &[999], &out).is_err());
}
