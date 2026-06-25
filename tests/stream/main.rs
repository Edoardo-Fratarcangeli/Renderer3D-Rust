//! Integration tests for the runtime streaming layer.
//!
//! Covers the in-process TCP server (`StreamSession`) end to end, with both a
//! pure-Rust producer and real **Python streamers** for the two wire formats:
//! NDJSON (always) and Arrow IPC (behind the `arrow-stream` feature). Python
//! tests skip gracefully when the interpreter / pyarrow is unavailable, so they
//! never fail CI environments that lack them.

mod helpers;
mod python_streamers;
mod rust_producer;
