//! Pure-Rust producer tests: no external dependencies, so they always run and
//! pin the `StreamSession` TCP server behavior precisely.

use std::io::Write;
use std::net::TcpStream;
use std::time::Duration;

use rendering_3d::dataset::preprocessor::ProjectionSpec;
use rendering_3d::dataset::stream::{StreamConfig, StreamFormat, StreamSession, StreamStatus};

use crate::helpers::{wait_for_rows, wait_until};

fn ndjson_config() -> StreamConfig {
    StreamConfig {
        format: StreamFormat::Ndjson,
        addr: "127.0.0.1:0".to_string(), // OS-assigned port
        max_rows: 10_000,
        projection: ProjectionSpec::default(),
    }
}

#[test]
fn ndjson_session_receives_rows_over_tcp() {
    let handle = StreamSession::start(ndjson_config()).unwrap();
    let addr = handle.addr().to_string();

    let mut sock = TcpStream::connect(&addr).expect("connect to stream server");
    for i in 0..64 {
        let line = format!("{{\"x\":[{},{}],\"label\":\"c{}\"}}\n", i, i + 1, i % 3);
        sock.write_all(line.as_bytes()).unwrap();
    }
    sock.flush().unwrap();

    assert!(
        wait_for_rows(&handle, 64, Duration::from_secs(5)),
        "did not receive 64 rows"
    );
    handle.with_buffer(|b| {
        assert_eq!(b.n_cols, 2);
        assert_eq!(b.n_rows(), 64);
        // Row 0 features and last row features.
        assert_eq!(&b.data[0..2], &[0.0, 1.0]);
        assert_eq!(&b.data[126..128], &[63.0, 64.0]);
    });
    let ds = handle.with_buffer(|b| b.to_dataset()).unwrap();
    assert_eq!(ds.label_names, vec!["c0", "c1", "c2"]);
}

#[test]
fn rolling_cap_keeps_only_newest_rows() {
    let mut cfg = ndjson_config();
    cfg.max_rows = 10;
    let handle = StreamSession::start(cfg).unwrap();
    let mut sock = TcpStream::connect(handle.addr()).unwrap();
    for i in 0..100 {
        writeln!(sock, "[{}]", i).unwrap();
    }
    sock.flush().unwrap();

    assert!(wait_for_rows(&handle, 100, Duration::from_secs(5)));
    handle.with_buffer(|b| {
        assert_eq!(b.n_rows(), 10, "rolling cap not enforced");
        assert_eq!(b.data.first().copied(), Some(90.0));
        assert_eq!(b.data.last().copied(), Some(99.0));
        assert_eq!(b.total_received, 100);
    });
}

#[test]
fn stop_is_responsive_and_marks_stopped() {
    let handle = StreamSession::start(ndjson_config()).unwrap();
    let _sock = TcpStream::connect(handle.addr()).unwrap();
    assert!(wait_until(Duration::from_secs(2), || handle.status()
        == StreamStatus::Active));
    let mut handle = handle;
    handle.stop();
    assert_eq!(handle.status(), StreamStatus::Stopped);
}

#[test]
fn malformed_line_sets_error_status() {
    let handle = StreamSession::start(ndjson_config()).unwrap();
    let mut sock = TcpStream::connect(handle.addr()).unwrap();
    sock.write_all(b"{not json}\n").unwrap();
    sock.flush().unwrap();
    assert!(
        wait_until(Duration::from_secs(5), || handle.status()
            == StreamStatus::Error),
        "malformed input should surface an error status"
    );
}

#[test]
fn supports_producer_reconnect() {
    let handle = StreamSession::start(ndjson_config()).unwrap();
    let addr = handle.addr().to_string();

    // First producer sends a few rows then disconnects.
    {
        let mut sock = TcpStream::connect(&addr).unwrap();
        for i in 0..5 {
            writeln!(sock, "[{}]", i).unwrap();
        }
        sock.flush().unwrap();
    }
    assert!(wait_for_rows(&handle, 5, Duration::from_secs(5)));

    // A second producer connects to the still-listening server.
    let mut sock2 = TcpStream::connect(&addr).unwrap();
    for i in 0..5 {
        writeln!(sock2, "[{}]", i + 100).unwrap();
    }
    sock2.flush().unwrap();
    assert!(wait_for_rows(&handle, 10, Duration::from_secs(5)));
}
