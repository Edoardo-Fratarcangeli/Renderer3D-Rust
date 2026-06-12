use std::collections::HashSet;

use rendering_3d::dataset::index::{
    apply_filter, CmpOp, DatasetIndex, FilterSpec, SearchQuery,
};
use rendering_3d::dataset::loader::{load, LoadOptions};

use crate::helpers;

fn sample_dataset(dir: &std::path::Path) -> rendering_3d::dataset::Dataset {
    let path = helpers::write_sample_npy(dir);
    load(&path, &LoadOptions::default()).unwrap()
}

#[test]
fn index_groups_rows_by_label() {
    let idx = DatasetIndex::build(&[0, 0, 1, 1, 2, 2], 3);
    assert_eq!(idx.n_rows, 6);
    assert_eq!(idx.count(0), 2);
    assert_eq!(idx.count(2), 2);
    assert_eq!(idx.label_rows[1], vec![2, 3]);
}

#[test]
fn index_persists_and_reloads() {
    let dir = tempfile::tempdir().unwrap();
    let idx = DatasetIndex::build(&[0, 1, 0, 1], 2);
    let path = dir.path().join("cache").join("test.index.json");
    idx.save_json(&path).unwrap();
    let reloaded = DatasetIndex::load_json(&path).unwrap();
    assert_eq!(idx, reloaded);
}

#[test]
fn label_filter_changes_visible_rows() {
    let dir = tempfile::tempdir().unwrap();
    let ds = sample_dataset(dir.path());
    let idx = DatasetIndex::build(&ds.labels, ds.label_names.len());

    // All labels enabled -> every row visible.
    let all = apply_filter(&ds, &idx, &FilterSpec::all_labels(3));
    assert_eq!(all, vec![0, 1, 2, 3, 4, 5]);

    // Only label "1" -> rows 2 and 3.
    let spec = FilterSpec {
        enabled_labels: HashSet::from([1]),
        query: SearchQuery::All,
    };
    assert_eq!(apply_filter(&ds, &idx, &spec), vec![2, 3]);

    // Nothing enabled -> empty selection.
    let spec = FilterSpec {
        enabled_labels: HashSet::new(),
        query: SearchQuery::All,
    };
    assert!(apply_filter(&ds, &idx, &spec).is_empty());
}

#[test]
fn search_query_parsing() {
    assert_eq!(SearchQuery::parse("").unwrap(), SearchQuery::All);
    assert_eq!(SearchQuery::parse("row:42").unwrap(), SearchQuery::Row(42));
    assert_eq!(
        SearchQuery::parse("c1 >= 5.0").unwrap(),
        SearchQuery::Column {
            col: 1,
            op: CmpOp::Ge,
            value: 5.0
        }
    );
    assert_eq!(
        SearchQuery::parse("Cluster_2").unwrap(),
        SearchQuery::LabelSubstring("cluster_2".into())
    );
    assert!(SearchQuery::parse("row:notanumber").is_err());
}

#[test]
fn search_narrows_selection() {
    let dir = tempfile::tempdir().unwrap();
    let ds = sample_dataset(dir.path());
    let idx = DatasetIndex::build(&ds.labels, ds.label_names.len());

    // Numeric predicate: first column > 4 -> rows 2..=5.
    let spec = FilterSpec {
        enabled_labels: (0..3).collect(),
        query: SearchQuery::parse("c0 > 4").unwrap(),
    };
    assert_eq!(apply_filter(&ds, &idx, &spec), vec![2, 3, 4, 5]);

    // Search composes with the label filter (intersection).
    let spec = FilterSpec {
        enabled_labels: HashSet::from([2]),
        query: SearchQuery::parse("c0 > 4").unwrap(),
    };
    assert_eq!(apply_filter(&ds, &idx, &spec), vec![4, 5]);

    // Row query.
    let spec = FilterSpec {
        enabled_labels: (0..3).collect(),
        query: SearchQuery::parse("row:3").unwrap(),
    };
    assert_eq!(apply_filter(&ds, &idx, &spec), vec![3]);

    // Label substring.
    let spec = FilterSpec {
        enabled_labels: (0..3).collect(),
        query: SearchQuery::parse("2").unwrap(),
    };
    assert_eq!(apply_filter(&ds, &idx, &spec), vec![4, 5]);
}
