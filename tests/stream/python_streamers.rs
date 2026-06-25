//! End-to-end tests driving the real Python streamers against the in-process
//! TCP server, one per wire format. They skip (and print a notice) when the
//! Python toolchain is missing, so they are safe on minimal CI images.

use std::process::Command;
use std::time::Duration;

use rendering_3d::dataset::preprocessor::ProjectionSpec;
use rendering_3d::dataset::stream::{StreamConfig, StreamFormat, StreamSession};

#[cfg(feature = "arrow-stream")]
use crate::helpers::pyarrow_available;
use crate::helpers::{python_available, script_path, split_addr, wait_for_rows};

/// Run a Python streamer script against a fresh session and assert the rows
/// decode to the deterministic values both scripts emit (`row i, col k = i*d+k`,
/// `label c{i%3}`).
fn run_streamer(format: StreamFormat, script: &str, n: usize, d: usize) {
    let cfg = StreamConfig {
        format,
        addr: "127.0.0.1:0".to_string(),
        max_rows: 100_000,
        projection: ProjectionSpec::default(),
    };
    let handle = StreamSession::start(cfg).unwrap();
    let (host, port) = split_addr(handle.addr());

    let status = Command::new("python3")
        .arg(script_path(script))
        .args([host, port, n.to_string(), d.to_string()])
        .status()
        .expect("spawn python3 streamer");
    assert!(status.success(), "python streamer exited with {status}");

    assert!(
        wait_for_rows(&handle, n as u64, Duration::from_secs(15)),
        "did not receive {n} rows from {script}"
    );

    handle.with_buffer(|b| {
        assert_eq!(b.n_cols, d);
        assert_eq!(b.n_rows(), n);
        // Spot-check a few decoded cells against the known formula.
        for &i in &[0usize, 1, n / 2, n - 1] {
            for k in 0..d {
                let got = b.data[i * d + k];
                assert_eq!(got, (i * d + k) as f32, "row {i} col {k}");
            }
        }
    });
    let ds = handle.with_buffer(|b| b.to_dataset()).unwrap();
    assert_eq!(ds.label_names, vec!["c0", "c1", "c2"]);
}

#[test]
fn python_ndjson_streamer_feeds_the_session() {
    if !python_available() {
        eprintln!("skipping: python3 not available");
        return;
    }
    run_streamer(StreamFormat::Ndjson, "ndjson_streamer.py", 300, 4);
}

#[cfg(feature = "arrow-stream")]
#[test]
fn python_arrow_streamer_feeds_the_session() {
    if !python_available() || !pyarrow_available() {
        eprintln!("skipping: python3 / pyarrow not available");
        return;
    }
    run_streamer(StreamFormat::ArrowIpc, "arrow_streamer.py", 500, 5);
}

#[cfg(feature = "arrow-stream")]
#[test]
fn python_arrow_multi_batch_streamer() {
    if !python_available() || !pyarrow_available() {
        eprintln!("skipping: python3 / pyarrow not available");
        return;
    }
    // Multiple record batches over one stream connection.
    let cfg = StreamConfig {
        format: StreamFormat::ArrowIpc,
        addr: "127.0.0.1:0".to_string(),
        max_rows: 100_000,
        projection: ProjectionSpec::default(),
    };
    let handle = StreamSession::start(cfg).unwrap();
    let (host, port) = split_addr(handle.addr());
    let status = Command::new("python3")
        .arg(script_path("arrow_streamer.py"))
        .args([host, port, "600".into(), "3".into(), "6".into()])
        .status()
        .expect("spawn python3 arrow streamer");
    assert!(status.success());
    assert!(wait_for_rows(&handle, 600, Duration::from_secs(15)));
    handle.with_buffer(|b| {
        assert_eq!(b.n_rows(), 600);
        assert_eq!(b.n_cols, 3);
    });
}
