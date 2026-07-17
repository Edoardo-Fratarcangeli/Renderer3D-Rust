//! Minimal Ollama REST client (raw `TcpStream` — no HTTP library dependency).
//!
//! Talks to a local Ollama daemon at [`OLLAMA_HOST`].
//!
//! Endpoints used:
//!  - `GET  /api/tags`    — list locally available models.
//!  - `POST /api/show`    — fetch architecture metadata for one model.
//!  - `POST /api/generate`— streaming NDJSON inference.
//!
//! All blocking calls run in background threads spawned by the caller.
//! The only synchronous function is [`list_models`]; [`load_model_graph`] and
//! [`run_inference`] are also synchronous but intended to be called inside
//! thread closures so they don't block the UI.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::mpsc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde_json::Value;

use crate::llm::arch::{ArchFamily, ArchSpec};
use crate::llm::network::NetworkGraph;
use crate::llm::tokenizer::Tokenizer;

pub const OLLAMA_HOST: &str = "127.0.0.1:11434";
const READ_TIMEOUT:  Duration = Duration::from_secs(120);
const WRITE_TIMEOUT: Duration = Duration::from_secs(10);

// The peer is only *nominally* Ollama: if the daemon is not running, any
// local process can squat port 11434, so sizes read off the wire are capped.
const MAX_CHUNK_BYTES: usize = 16 * 1024 * 1024;
const MAX_BODY_BYTES:  usize = 64 * 1024 * 1024;

// ─── Public types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct OllamaModel {
    pub name: String,
    pub size_bytes: u64,
    pub family: String,
    pub parameter_size: String,
}

/// Event stream from [`run_inference`].
pub enum OllamaEvent {
    /// One decoded token string from the model.
    Token(String),
    /// Inference finished successfully.
    Done,
    /// Fatal error; inference cannot continue.
    Error(String),
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Fetch the list of locally-installed Ollama models.
pub fn list_models() -> Result<Vec<OllamaModel>> {
    let body = http_get("/api/tags")?;
    let json: Value = serde_json::from_str(&body).context("Ollama /api/tags: invalid JSON")?;
    let arr = json["models"].as_array().context("No 'models' array in /api/tags response")?;
    Ok(arr
        .iter()
        .map(|m| OllamaModel {
            name:           m["name"].as_str().unwrap_or("?").to_owned(),
            size_bytes:     m["size"].as_u64().unwrap_or(0),
            family:         m["details"]["family"].as_str().unwrap_or("").to_owned(),
            parameter_size: m["details"]["parameter_size"].as_str().unwrap_or("?").to_owned(),
        })
        .collect())
}

/// Fetch architecture metadata for `model_name` and build a [`NetworkGraph`].
///
/// Uses `/api/show` to retrieve `model_info` (same key convention as GGUF KV).
/// Returns `(graph, None)` — Ollama's show endpoint does not expose vocabulary.
pub fn load_model_graph(model_name: &str) -> Result<(NetworkGraph, Option<Tokenizer>)> {
    let payload = serde_json::json!({ "name": model_name }).to_string();
    let body    = http_post("/api/show", &payload)?;
    let json: Value = serde_json::from_str(&body).context("Ollama /api/show: invalid JSON")?;

    let arch_str = json["model_info"]["general.architecture"]
        .as_str()
        .or_else(|| json["details"]["family"].as_str())
        .unwrap_or("llama")
        .to_owned();

    let family = ArchFamily::detect(&arch_str, model_name);

    // Collect all numeric entries from model_info (keeps full "arch.param" keys).
    let mut meta: HashMap<String, u64> = HashMap::new();
    if let Some(obj) = json["model_info"].as_object() {
        for (k, v) in obj {
            if let Some(n) = v.as_u64() {
                meta.insert(k.clone(), n);
            }
        }
    }

    let spec = if meta.values().any(|_| true) {
        ArchSpec::from_metadata(family, &arch_str, &meta)
    } else {
        ArchSpec::default_for(family)
    };

    let (layers, vram_gb, moe_config) = crate::llm::arch::build_layers(&spec);
    let mut graph = NetworkGraph {
        name:   format!("{model_name} (Ollama)"),
        layers,
        edges:  vec![],
        estimated_vram_gb: Some(vram_gb),
        moe_config,
        data_graph: false,
        note: None,
    };
    graph.layout();
    Ok((graph, None))
}

/// Spawn an inference thread and return the event [`mpsc::Receiver`].
///
/// Each [`OllamaEvent::Token`] carries one decoded fragment;
/// [`OllamaEvent::Done`] signals end-of-stream.
pub fn run_inference(model: &str, prompt: &str) -> mpsc::Receiver<OllamaEvent> {
    let (tx, rx) = mpsc::channel();
    let model  = model.to_owned();
    let prompt = prompt.to_owned();
    std::thread::spawn(move || {
        if let Err(e) = do_inference(&model, &prompt, &tx) {
            let _ = tx.send(OllamaEvent::Error(e.to_string()));
        }
    });
    rx
}

// ─── HTTP helpers ─────────────────────────────────────────────────────────────

fn open_stream() -> Result<TcpStream> {
    let stream = TcpStream::connect(OLLAMA_HOST)
        .with_context(|| format!("Cannot reach Ollama at {OLLAMA_HOST} — is it running? (ollama serve)"))?;
    stream.set_read_timeout(Some(READ_TIMEOUT))?;
    stream.set_write_timeout(Some(WRITE_TIMEOUT))?;
    Ok(stream)
}

fn http_get(path: &str) -> Result<String> {
    let mut stream = open_stream()?;
    write!(stream,
        "GET {path} HTTP/1.1\r\nHost: {OLLAMA_HOST}\r\nConnection: close\r\n\r\n")?;
    stream.flush()?;
    drain_response(BufReader::new(stream))
}

fn http_post(path: &str, body: &str) -> Result<String> {
    let mut stream = open_stream()?;
    write!(stream,
        "POST {path} HTTP/1.1\r\nHost: {OLLAMA_HOST}\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n{body}",
        body.len())?;
    stream.flush()?;
    drain_response(BufReader::new(stream))
}

/// Read past headers; return decoded response body as a String.
fn drain_response<R: Read + BufRead>(mut reader: R) -> Result<String> {
    let (status, is_chunked) = parse_headers(&mut reader)?;
    if status != 200 {
        let err = read_capped_string(&mut reader)?;
        bail!("Ollama HTTP {status}: {}", err.chars().take(300).collect::<String>());
    }
    if is_chunked {
        read_chunked(&mut reader)
    } else {
        read_capped_string(&mut reader)
    }
}

/// `read_to_string` bounded by [`MAX_BODY_BYTES`] (bodies come off the wire).
fn read_capped_string<R: Read>(reader: &mut R) -> Result<String> {
    let mut body = String::new();
    reader.take(MAX_BODY_BYTES as u64 + 1).read_to_string(&mut body)?;
    if body.len() > MAX_BODY_BYTES {
        bail!("HTTP response body exceeds the {MAX_BODY_BYTES} byte cap");
    }
    Ok(body)
}

/// Returns (HTTP status code, is_chunked).
fn parse_headers<R: BufRead>(reader: &mut R) -> Result<(u16, bool)> {
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let status: u16 = line.split_whitespace().nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mut chunked = false;
    loop {
        line.clear();
        reader.read_line(&mut line)?;
        if line.trim().is_empty() { break; }
        if line.to_ascii_lowercase().contains("transfer-encoding: chunked") {
            chunked = true;
        }
    }
    Ok((status, chunked))
}

/// Parse one chunked-encoding size line. Malformed hex is an ERROR — the old
/// lenient `unwrap_or(0)` treated it as end-of-stream, silently truncating
/// the data. Sizes above [`MAX_CHUNK_BYTES`] are rejected before allocating.
fn parse_chunk_size(sz_line: &str) -> Result<usize> {
    let hex = sz_line.trim().split(';').next().unwrap_or("").trim();
    if hex.is_empty() || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        bail!("malformed HTTP chunk size line: {:?}", sz_line.trim());
    }
    let size = usize::from_str_radix(hex, 16)
        .with_context(|| format!("HTTP chunk size out of range: {hex:?}"))?;
    if size > MAX_CHUNK_BYTES {
        bail!("HTTP chunk of {size} bytes exceeds the {MAX_CHUNK_BYTES} byte cap");
    }
    Ok(size)
}

/// Append one chunk body to `out`, reading in fixed 16 KiB blocks so the
/// declared size is never allocated upfront; consumes the trailing CRLF.
fn read_chunk_into<R: BufRead>(reader: &mut R, size: usize, out: &mut Vec<u8>) -> Result<()> {
    const BLOCK: usize = 16 * 1024;
    let mut block = [0u8; BLOCK];
    let mut remaining = size;
    while remaining > 0 {
        let n = remaining.min(BLOCK);
        reader.read_exact(&mut block[..n])?;
        out.extend_from_slice(&block[..n]);
        remaining -= n;
    }
    let mut crlf = [0u8; 2];
    let _ = reader.read_exact(&mut crlf);
    Ok(())
}

fn read_chunked<R: BufRead>(reader: &mut R) -> Result<String> {
    let mut buf: Vec<u8> = Vec::new();
    loop {
        let mut sz_line = String::new();
        reader.read_line(&mut sz_line)?;
        let chunk_size = parse_chunk_size(&sz_line)?;
        if chunk_size == 0 { break; }
        if buf.len() + chunk_size > MAX_BODY_BYTES {
            bail!("HTTP response body exceeds the {MAX_BODY_BYTES} byte cap");
        }
        read_chunk_into(reader, chunk_size, &mut buf)?;
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

// ─── Streaming inference ──────────────────────────────────────────────────────

fn do_inference(
    model: &str,
    prompt: &str,
    tx: &mpsc::Sender<OllamaEvent>,
) -> Result<()> {
    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": true
    })
    .to_string();

    let mut stream = open_stream()?;
    write!(stream,
        "POST /api/generate HTTP/1.1\r\nHost: {OLLAMA_HOST}\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n{body}",
        body.len())?;
    stream.flush()?;

    let mut reader = BufReader::new(stream);
    let (status, is_chunked) = parse_headers(&mut reader)?;
    if status != 200 {
        let err = read_capped_string(&mut reader)?;
        bail!("Ollama /api/generate HTTP {status}: {}",
              err.chars().take(300).collect::<String>());
    }

    if is_chunked {
        stream_chunked_ndjson(&mut reader, tx)
    } else {
        stream_plain_ndjson(&mut reader, tx)
    }
}

fn stream_plain_ndjson<R: BufRead>(
    reader: &mut R,
    tx: &mpsc::Sender<OllamaEvent>,
) -> Result<()> {
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 { break; }
        if line.len() > MAX_CHUNK_BYTES {
            bail!("NDJSON line exceeds the {MAX_CHUNK_BYTES} byte cap");
        }
        dispatch_ndjson(line.trim(), tx);
    }
    Ok(())
}

fn stream_chunked_ndjson<R: BufRead>(
    reader: &mut R,
    tx: &mpsc::Sender<OllamaEvent>,
) -> Result<()> {
    let mut carry = String::new();
    loop {
        let mut sz_line = String::new();
        reader.read_line(&mut sz_line)?;
        let chunk_size = parse_chunk_size(&sz_line)?;
        if chunk_size == 0 { break; }
        let mut chunk = Vec::new();
        read_chunk_into(reader, chunk_size, &mut chunk)?;
        if carry.len() + chunk.len() > MAX_BODY_BYTES {
            bail!("NDJSON carry buffer exceeds the {MAX_BODY_BYTES} byte cap");
        }

        let text = String::from_utf8_lossy(&chunk);
        // Prepend any partial JSON fragment from the previous chunk.
        let combined = format!("{carry}{text}");
        carry.clear();

        // Split on newlines; last entry may be an incomplete JSON line.
        let mut parts = combined.split('\n').peekable();
        while let Some(part) = parts.next() {
            if parts.peek().is_none() && !part.trim_end().ends_with('}') {
                carry = part.to_owned();
            } else {
                dispatch_ndjson(part.trim(), tx);
            }
        }
    }
    if !carry.trim().is_empty() {
        dispatch_ndjson(carry.trim(), tx);
    }
    Ok(())
}

fn dispatch_ndjson(s: &str, tx: &mpsc::Sender<OllamaEvent>) {
    if s.is_empty() { return; }
    if let Ok(obj) = serde_json::from_str::<Value>(s) {
        if let Some(tok) = obj["response"].as_str() {
            if !tok.is_empty() {
                let _ = tx.send(OllamaEvent::Token(tok.to_owned()));
            }
        }
        if obj["done"].as_bool() == Some(true) {
            let _ = tx.send(OllamaEvent::Done);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_size_parses_valid_hex_and_extensions() {
        assert_eq!(parse_chunk_size("1a\r\n").unwrap(), 0x1a);
        assert_eq!(parse_chunk_size("0\r\n").unwrap(), 0);
        assert_eq!(parse_chunk_size("ff;ext=1\r\n").unwrap(), 0xff);
    }

    #[test]
    fn chunk_size_rejects_malformed_and_oversized() {
        // Malformed hex must be an error, not a silent end-of-stream.
        assert!(parse_chunk_size("zz\r\n").is_err());
        assert!(parse_chunk_size("\r\n").is_err());
        // A fake server declaring a 4 GiB chunk must not cause an allocation.
        assert!(parse_chunk_size("FFFFFFFF\r\n").is_err());
    }

    #[test]
    fn chunked_body_is_reassembled_and_capped() {
        let wire = b"5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
        let body = read_chunked(&mut std::io::Cursor::new(&wire[..])).unwrap();
        assert_eq!(body, "hello world");

        let evil = b"FFFFFFFF\r\n";
        assert!(read_chunked(&mut std::io::Cursor::new(&evil[..])).is_err());
    }
}
