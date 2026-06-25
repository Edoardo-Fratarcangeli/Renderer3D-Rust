//! Shared helpers for the streaming integration tests.

use std::time::{Duration, Instant};

use rendering_3d::dataset::stream::StreamHandle;

/// Poll `cond` until it returns true or `timeout` elapses.
pub fn wait_until(timeout: Duration, mut cond: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    cond()
}

/// Block until the handle has received at least `n` rows (or time out).
pub fn wait_for_rows(handle: &StreamHandle, n: u64, timeout: Duration) -> bool {
    wait_until(timeout, || handle.total_received() >= n)
}

/// Split a `host:port` address into its parts for passing to a subprocess.
pub fn split_addr(addr: &str) -> (String, String) {
    let (host, port) = addr.rsplit_once(':').expect("addr must be host:port");
    (host.to_string(), port.to_string())
}

/// True when `python3` can be executed at all.
pub fn python_available() -> bool {
    std::process::Command::new("python3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// True when `import pyarrow` succeeds.
#[cfg_attr(not(feature = "arrow-stream"), allow(dead_code))]
pub fn pyarrow_available() -> bool {
    std::process::Command::new("python3")
        .args(["-c", "import pyarrow"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Absolute path to a Python streamer script in this test directory.
pub fn script_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("stream")
        .join(name)
}
