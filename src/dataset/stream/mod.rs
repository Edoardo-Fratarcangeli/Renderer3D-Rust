//! Runtime data streaming: ingest rows from an external producer (a Python
//! process, another language, …) while the app is running and feed them into
//! the **same** projection + render pipeline used by file import.
//!
//! ## Wire formats (Strategy pattern — [`BatchDecoder`])
//! - **NDJSON** (newline-delimited JSON) over TCP — maximum compatibility,
//!   zero extra dependencies: any language can `socket.send` one JSON row per
//!   line.
//! - **Arrow IPC stream** (behind the optional `arrow-stream` feature) —
//!   columnar, high-throughput decoding.
//!
//! ## Architecture (no egui / wgpu here — fully unit-testable)
//! ```text
//! producer ──TCP──▶ StreamSession (background thread)
//!                     │  accept + BatchDecoder::next_batch
//!                     ▼
//!                  StreamShared { Mutex<StreamBuffer>, status, last_rx, stop }
//!                     ▲                         │ lifecycle
//!   UI polls version ─┘                         ▼  StreamEvent (mpsc)
//! ```
//! The app acts as a **TCP server**: it listens and the producer connects, so
//! it survives producer restarts (the session keeps accepting new connections).
//! [`StreamHandle`] is the RAII front-end held by the UI; dropping it stops the
//! thread.

#[cfg(feature = "arrow-stream")]
mod arrow;
mod ndjson;

pub use ndjson::NdjsonDecoder;

use std::io::Read;
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::loader::labels_from_strings;
use super::metadata::DatasetMetadata;
use super::preprocessor::ProjectionSpec;
use super::{Dataset, DatasetError, FeatureSource, Result};

/// How long the accept loop sleeps between polls while waiting for a peer.
const ACCEPT_POLL: Duration = Duration::from_millis(20);

/// Wire format of the incoming stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamFormat {
    /// Newline-delimited JSON (one row object per line).
    Ndjson,
    /// Apache Arrow IPC stream (feature `arrow-stream`).
    ArrowIpc,
}

impl StreamFormat {
    pub fn tag(&self) -> &'static str {
        match self {
            StreamFormat::Ndjson => "ndjson",
            StreamFormat::ArrowIpc => "arrow",
        }
    }
}

/// Runtime status of a stream, mirrored in an atomic so the UI can read it
/// every frame without locking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamStatus {
    /// Not started.
    Idle,
    /// Listening / connected and waiting for or processing data.
    Active,
    /// The session ended with an error.
    Error,
    /// Cleanly stopped by the user.
    Stopped,
}

impl StreamStatus {
    fn to_u8(self) -> u8 {
        match self {
            StreamStatus::Idle => 0,
            StreamStatus::Active => 1,
            StreamStatus::Error => 2,
            StreamStatus::Stopped => 3,
        }
    }
    fn from_u8(v: u8) -> Self {
        match v {
            1 => StreamStatus::Active,
            2 => StreamStatus::Error,
            3 => StreamStatus::Stopped,
            _ => StreamStatus::Idle,
        }
    }
}

/// Lifecycle notifications surfaced to the UI as status text.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// The server is bound and listening on this address.
    Listening(String),
    /// A producer connected from this peer address.
    Connected(String),
    /// The current connection ended cleanly (peer closed / EOF).
    Disconnected,
    /// The session failed; carries a human-readable message.
    Error(String),
}

/// Configuration for a streaming session.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub format: StreamFormat,
    /// Bind address, e.g. `"127.0.0.1:8765"`. Use port `0` for an OS-assigned
    /// port (the actual address is reported via [`StreamEvent::Listening`]).
    pub addr: String,
    /// Maximum rows kept in the rolling buffer (oldest are dropped).
    pub max_rows: usize,
    /// Projection applied to the rolling buffer for the 3D preview.
    pub projection: ProjectionSpec,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            format: StreamFormat::Ndjson,
            addr: "127.0.0.1:8765".to_string(),
            max_rows: 50_000,
            projection: ProjectionSpec::default(),
        }
    }
}

/// One decoded chunk of rows. NDJSON yields one row per call; Arrow yields a
/// whole record batch.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RowBatch {
    /// Feature columns per row (must be constant across a session).
    pub n_cols: usize,
    /// Row-major feature values, length `n_rows * n_cols`.
    pub rows: Vec<f32>,
    /// Per-row label strings; empty when the stream carries no labels.
    pub labels: Vec<String>,
}

impl RowBatch {
    pub fn n_rows(&self) -> usize {
        if self.n_cols == 0 {
            0
        } else {
            self.rows.len() / self.n_cols
        }
    }
}

/// Strategy for turning a byte stream into [`RowBatch`]es.
pub trait BatchDecoder: Send {
    /// Next batch, or `Ok(None)` at clean end of stream.
    fn next_batch(&mut self) -> Result<Option<RowBatch>>;
}

/// Rolling, bounded accumulator of streamed rows shared between the worker
/// thread and the UI.
#[derive(Debug)]
pub struct StreamBuffer {
    pub n_cols: usize,
    pub max_rows: usize,
    /// Row-major features for the rows currently retained.
    pub data: Vec<f32>,
    /// Per-row labels (parallel to rows); empty strings when unlabeled.
    pub labels: Vec<String>,
    /// Whether any batch carried labels.
    pub has_labels: bool,
    /// Bumped on every successful append, so the UI can detect changes.
    pub version: u64,
    /// Monotonic count of rows received over the whole session.
    pub total_received: u64,
}

impl StreamBuffer {
    pub fn new(max_rows: usize) -> Self {
        Self {
            n_cols: 0,
            max_rows: max_rows.max(1),
            data: Vec::new(),
            labels: Vec::new(),
            has_labels: false,
            version: 0,
            total_received: 0,
        }
    }

    pub fn n_rows(&self) -> usize {
        if self.n_cols == 0 {
            0
        } else {
            self.data.len() / self.n_cols
        }
    }

    /// Append a decoded batch, enforcing a constant column count and the
    /// rolling row cap (oldest rows are dropped).
    pub fn push_batch(&mut self, batch: &RowBatch) -> Result<()> {
        if batch.n_rows() == 0 {
            return Ok(());
        }
        if self.n_cols == 0 {
            self.n_cols = batch.n_cols;
            self.has_labels = !batch.labels.is_empty();
        } else if batch.n_cols != self.n_cols {
            return Err(DatasetError::Format(format!(
                "stream column count changed from {} to {}",
                self.n_cols, batch.n_cols
            )));
        }

        self.data.extend_from_slice(&batch.rows);
        if self.has_labels {
            if batch.labels.len() == batch.n_rows() {
                self.labels.extend_from_slice(&batch.labels);
            } else {
                // Keep rows/labels aligned even if a later batch omits labels.
                self.labels
                    .extend(std::iter::repeat_n(String::new(), batch.n_rows()));
            }
        }
        self.total_received += batch.n_rows() as u64;

        // Enforce the rolling cap.
        let n_rows = self.n_rows();
        if n_rows > self.max_rows {
            let drop = n_rows - self.max_rows;
            self.data.drain(0..drop * self.n_cols);
            if self.has_labels && self.labels.len() >= drop {
                self.labels.drain(0..drop);
            }
        }
        self.version = self.version.wrapping_add(1);
        Ok(())
    }

    pub fn clear(&mut self) {
        self.n_cols = 0;
        self.data.clear();
        self.labels.clear();
        self.has_labels = false;
        self.version = self.version.wrapping_add(1);
        self.total_received = 0;
    }

    /// Build an in-memory [`Dataset`] snapshot of the current buffer, ready for
    /// the standard projection pipeline. Returns `None` while empty.
    pub fn to_dataset(&self) -> Option<Dataset> {
        let n_rows = self.n_rows();
        if n_rows == 0 {
            return None;
        }
        let (labels, label_names) = if self.has_labels && self.labels.len() == n_rows {
            labels_from_strings(&self.labels)
        } else {
            (vec![0u32; n_rows], vec!["unlabeled".to_string()])
        };
        let mut metadata = DatasetMetadata::new("stream", "stream", n_rows, self.n_cols);
        metadata.column_names = (0..self.n_cols).map(|i| format!("f{i}")).collect();
        metadata.set_label_stats(&labels, &label_names);
        Some(Dataset {
            metadata,
            source: FeatureSource::InMemory(self.data.clone()),
            labels,
            label_names,
        })
    }
}

/// State shared between the worker thread and the UI.
#[derive(Debug)]
pub struct StreamShared {
    pub buffer: Mutex<StreamBuffer>,
    status: AtomicU8,
    /// Epoch milliseconds of the last received batch (0 = none yet).
    last_rx_ms: AtomicU64,
    stop: AtomicBool,
    /// The live connection, kept so [`StreamHandle::stop`] can unblock a
    /// blocking read by shutting the socket down.
    conn: Mutex<Option<TcpStream>>,
}

impl StreamShared {
    fn set_status(&self, s: StreamStatus) {
        self.status.store(s.to_u8(), Ordering::SeqCst);
    }
    pub fn status(&self) -> StreamStatus {
        StreamStatus::from_u8(self.status.load(Ordering::SeqCst))
    }
    fn mark_received(&self) {
        self.last_rx_ms.store(now_ms(), Ordering::SeqCst);
    }
    /// True when a batch arrived within `window`.
    pub fn receiving_within(&self, window: Duration) -> bool {
        let last = self.last_rx_ms.load(Ordering::SeqCst);
        last != 0 && now_ms().saturating_sub(last) <= window.as_millis() as u64
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Build the decoder for a format over an owned TCP stream.
fn build_decoder(format: StreamFormat, stream: TcpStream) -> Result<Box<dyn BatchDecoder>> {
    match format {
        StreamFormat::Ndjson => Ok(Box::new(NdjsonDecoder::new(stream))),
        #[cfg(feature = "arrow-stream")]
        StreamFormat::ArrowIpc => Ok(Box::new(arrow::ArrowIpcDecoder::new(stream)?)),
        #[cfg(not(feature = "arrow-stream"))]
        StreamFormat::ArrowIpc => Err(DatasetError::Unsupported(
            "Arrow IPC streaming not compiled in (enable feature `arrow-stream`)".into(),
        )),
    }
}

/// Owns the background streaming thread and the shared state.
pub struct StreamSession;

impl StreamSession {
    /// Bind the listener (surfacing bind errors immediately) and spawn the
    /// accept/decode loop. Returns the RAII [`StreamHandle`].
    pub fn start(config: StreamConfig) -> Result<StreamHandle> {
        let listener = TcpListener::bind(&config.addr)?;
        listener.set_nonblocking(true)?;
        let local = listener
            .local_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|_| config.addr.clone());

        let shared = Arc::new(StreamShared {
            buffer: Mutex::new(StreamBuffer::new(config.max_rows)),
            status: AtomicU8::new(StreamStatus::Active.to_u8()),
            last_rx_ms: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            conn: Mutex::new(None),
        });
        let (tx, rx) = channel();
        let _ = tx.send(StreamEvent::Listening(local.clone()));

        let worker_shared = Arc::clone(&shared);
        let format = config.format;
        let join = std::thread::spawn(move || {
            run_session(listener, format, worker_shared, tx);
        });

        Ok(StreamHandle {
            shared,
            events: rx,
            join: Some(join),
            addr: local,
        })
    }
}

/// The worker loop: accept connections and decode batches until stopped.
fn run_session(
    listener: TcpListener,
    format: StreamFormat,
    shared: Arc<StreamShared>,
    tx: Sender<StreamEvent>,
) {
    'accept: loop {
        if shared.stop.load(Ordering::SeqCst) {
            break;
        }
        let (stream, peer) = match listener.accept() {
            Ok(pair) => pair,
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(ACCEPT_POLL);
                continue;
            }
            Err(e) => {
                let _ = tx.send(StreamEvent::Error(format!("accept failed: {e}")));
                shared.set_status(StreamStatus::Error);
                break;
            }
        };
        let _ = tx.send(StreamEvent::Connected(peer.to_string()));

        // Blocking reads from here on; stop() unblocks them via shutdown().
        let _ = stream.set_nonblocking(false);
        if let Ok(clone) = stream.try_clone() {
            *shared.conn.lock().unwrap() = Some(clone);
        }
        let mut decoder = match build_decoder(format, stream) {
            Ok(d) => d,
            Err(e) => {
                let _ = tx.send(StreamEvent::Error(e.to_string()));
                shared.set_status(StreamStatus::Error);
                break;
            }
        };

        loop {
            if shared.stop.load(Ordering::SeqCst) {
                break 'accept;
            }
            match decoder.next_batch() {
                Ok(Some(batch)) => {
                    let mut buf = shared.buffer.lock().unwrap();
                    if let Err(e) = buf.push_batch(&batch) {
                        drop(buf);
                        let _ = tx.send(StreamEvent::Error(e.to_string()));
                        shared.set_status(StreamStatus::Error);
                        break 'accept;
                    }
                    drop(buf);
                    shared.mark_received();
                }
                Ok(None) => {
                    let _ = tx.send(StreamEvent::Disconnected);
                    break; // back to accept: support producer reconnects
                }
                Err(_) if shared.stop.load(Ordering::SeqCst) => break 'accept,
                Err(e) => {
                    let _ = tx.send(StreamEvent::Error(e.to_string()));
                    shared.set_status(StreamStatus::Error);
                    break;
                }
            }
        }
        *shared.conn.lock().unwrap() = None;
    }

    // Preserve an explicit Error status; otherwise mark a clean stop.
    if shared.status() != StreamStatus::Error {
        shared.set_status(StreamStatus::Stopped);
    }
}

/// RAII front-end to a streaming session held by the UI. Dropping it stops the
/// worker thread.
pub struct StreamHandle {
    shared: Arc<StreamShared>,
    events: Receiver<StreamEvent>,
    join: Option<JoinHandle<()>>,
    addr: String,
}

impl StreamHandle {
    /// Resolved bound address (with the real port when `0` was requested).
    pub fn addr(&self) -> &str {
        &self.addr
    }

    pub fn status(&self) -> StreamStatus {
        self.shared.status()
    }

    /// True when data arrived within `window` (drives the "receiving" color).
    pub fn is_receiving(&self, window: Duration) -> bool {
        self.shared.status() == StreamStatus::Active && self.shared.receiving_within(window)
    }

    /// Current buffer version (changes whenever rows are appended/cleared).
    pub fn buffer_version(&self) -> u64 {
        self.shared.buffer.lock().unwrap().version
    }

    pub fn total_received(&self) -> u64 {
        self.shared.buffer.lock().unwrap().total_received
    }

    /// Run `f` with shared read access to the buffer.
    pub fn with_buffer<R>(&self, f: impl FnOnce(&StreamBuffer) -> R) -> R {
        f(&self.shared.buffer.lock().unwrap())
    }

    /// Drain pending lifecycle events (non-blocking).
    pub fn drain_events(&self) -> Vec<StreamEvent> {
        self.events.try_iter().collect()
    }

    /// Stop the worker thread and wait for it to finish. Idempotent.
    pub fn stop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        if let Some(conn) = self.shared.conn.lock().unwrap().take() {
            let _ = conn.shutdown(std::net::Shutdown::Both);
        }
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl Drop for StreamHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Parse a single JSON value (object or array) into a one-row [`RowBatch`].
/// Shared by the NDJSON decoder and its tests.
pub(crate) fn parse_json_row(value: &serde_json::Value) -> Result<RowBatch> {
    use serde_json::Value;

    // Feature vector: a bare array, or the first of x/features/data/values.
    let (features, label): (&Vec<Value>, Option<String>) = match value {
        Value::Array(arr) => (arr, None),
        Value::Object(map) => {
            let feats = ["x", "features", "data", "values"]
                .iter()
                .find_map(|k| map.get(*k))
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    DatasetError::Format(
                        "NDJSON row object needs an array field (x/features/data/values)".into(),
                    )
                })?;
            let label = ["label", "labels", "target", "class", "y"]
                .iter()
                .find_map(|k| map.get(*k))
                .map(json_scalar_to_string);
            (feats, label)
        }
        _ => {
            return Err(DatasetError::Format(
                "NDJSON row must be a JSON array or object".into(),
            ))
        }
    };

    let mut rows = Vec::with_capacity(features.len());
    for v in features {
        let n = v
            .as_f64()
            .ok_or_else(|| DatasetError::Format("NDJSON feature values must be numbers".into()))?;
        rows.push(n as f32);
    }
    Ok(RowBatch {
        n_cols: rows.len(),
        rows,
        labels: label.into_iter().collect(),
    })
}

fn json_scalar_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Test-only blocking read helper used by the integration suite. Kept here so
/// the worker plumbing is reachable without exposing internals.
#[doc(hidden)]
pub fn decode_all<R: Read + Send + 'static>(
    format: StreamFormat,
    reader: R,
    out: &mut StreamBuffer,
) -> Result<()> {
    let mut decoder: Box<dyn BatchDecoder> = match format {
        StreamFormat::Ndjson => Box::new(NdjsonDecoder::new(reader)),
        #[cfg(feature = "arrow-stream")]
        StreamFormat::ArrowIpc => Box::new(arrow::ArrowIpcDecoder::new(reader)?),
        #[cfg(not(feature = "arrow-stream"))]
        StreamFormat::ArrowIpc => {
            return Err(DatasetError::Unsupported("arrow-stream feature off".into()))
        }
    };
    while let Some(batch) = decoder.next_batch()? {
        out.push_batch(&batch)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_object_row_with_label() {
        let v = serde_json::json!({"x": [1.0, 2.0, 3.0], "label": "cat"});
        let b = parse_json_row(&v).unwrap();
        assert_eq!(b.n_cols, 3);
        assert_eq!(b.rows, vec![1.0, 2.0, 3.0]);
        assert_eq!(b.labels, vec!["cat".to_string()]);
    }

    #[test]
    fn parse_bare_array_row_is_unlabeled() {
        let v = serde_json::json!([4.0, 5.0]);
        let b = parse_json_row(&v).unwrap();
        assert_eq!(b.rows, vec![4.0, 5.0]);
        assert!(b.labels.is_empty());
    }

    #[test]
    fn parse_numeric_label_is_stringified() {
        let v = serde_json::json!({"features": [1.0], "y": 7});
        let b = parse_json_row(&v).unwrap();
        assert_eq!(b.labels, vec!["7".to_string()]);
    }

    #[test]
    fn parse_rejects_non_numeric_and_missing_fields() {
        assert!(parse_json_row(&serde_json::json!({"x": ["a", "b"]})).is_err());
        assert!(parse_json_row(&serde_json::json!({"nope": 1})).is_err());
        assert!(parse_json_row(&serde_json::json!(42)).is_err());
    }

    #[test]
    fn buffer_enforces_constant_columns() {
        let mut buf = StreamBuffer::new(10);
        buf.push_batch(&RowBatch {
            n_cols: 2,
            rows: vec![1.0, 2.0],
            labels: vec![],
        })
        .unwrap();
        let err = buf.push_batch(&RowBatch {
            n_cols: 3,
            rows: vec![1.0, 2.0, 3.0],
            labels: vec![],
        });
        assert!(err.is_err());
    }

    #[test]
    fn buffer_rolls_oldest_rows_off_the_cap() {
        let mut buf = StreamBuffer::new(2);
        for i in 0..5 {
            buf.push_batch(&RowBatch {
                n_cols: 1,
                rows: vec![i as f32],
                labels: vec![],
            })
            .unwrap();
        }
        assert_eq!(buf.n_rows(), 2);
        assert_eq!(buf.data, vec![3.0, 4.0]); // newest two retained
        assert_eq!(buf.total_received, 5);
        assert!(buf.version >= 5);
    }

    #[test]
    fn buffer_keeps_labels_aligned_and_builds_dataset() {
        let mut buf = StreamBuffer::new(10);
        for i in 0..3 {
            buf.push_batch(&RowBatch {
                n_cols: 2,
                rows: vec![i as f32, i as f32 + 0.5],
                labels: vec![format!("c{}", i % 2)],
            })
            .unwrap();
        }
        let ds = buf.to_dataset().unwrap();
        assert_eq!(ds.n_rows(), 3);
        assert_eq!(ds.n_cols(), 2);
        assert_eq!(ds.label_names, vec!["c0".to_string(), "c1".to_string()]);
    }

    #[test]
    fn empty_buffer_has_no_dataset() {
        assert!(StreamBuffer::new(4).to_dataset().is_none());
    }

    #[test]
    fn status_roundtrips_through_u8() {
        for s in [
            StreamStatus::Idle,
            StreamStatus::Active,
            StreamStatus::Error,
            StreamStatus::Stopped,
        ] {
            assert_eq!(StreamStatus::from_u8(s.to_u8()), s);
        }
    }
}
