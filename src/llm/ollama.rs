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

    let (layers, vram_gb) = crate::llm::arch::build_layers(&spec);
    let mut graph = NetworkGraph {
        name:   format!("{model_name} (Ollama)"),
        layers,
        edges:  vec![],
        estimated_vram_gb: Some(vram_gb),
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
        let mut err = String::new();
        reader.read_to_string(&mut err)?;
        bail!("Ollama HTTP {status}: {}", err.chars().take(300).collect::<String>());
    }
    if is_chunked {
        read_chunked(&mut reader)
    } else {
        let mut body = String::new();
        reader.read_to_string(&mut body)?;
        Ok(body)
    }
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

fn read_chunked<R: BufRead>(reader: &mut R) -> Result<String> {
    let mut buf: Vec<u8> = Vec::new();
    loop {
        let mut sz_line = String::new();
        reader.read_line(&mut sz_line)?;
        let hex = sz_line.trim().split(';').next().unwrap_or("0");
        let chunk_size = usize::from_str_radix(hex, 16).unwrap_or(0);
        if chunk_size == 0 { break; }
        let mut chunk = vec![0u8; chunk_size];
        reader.read_exact(&mut chunk)?;
        buf.extend_from_slice(&chunk);
        let mut crlf = [0u8; 2];
        let _ = reader.read_exact(&mut crlf);
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
        let mut err = String::new();
        reader.read_to_string(&mut err)?;
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
        let hex = sz_line.trim().split(';').next().unwrap_or("0");
        let chunk_size = usize::from_str_radix(hex, 16).unwrap_or(0);
        if chunk_size == 0 { break; }
        let mut chunk = vec![0u8; chunk_size];
        reader.read_exact(&mut chunk)?;
        let mut crlf = [0u8; 2];
        let _ = reader.read_exact(&mut crlf);

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
