//! Minimal whitespace tokenizer backed by a vocabulary list.
//!
//! For models loaded with an embedded vocabulary (GGUF `tokenizer.ggml.tokens`
//! or JSON `vocabulary`), exact word lookup is used. Unknown words fall back
//! to a djb2-hash bucket so the visualization is always populated.

use std::collections::HashMap;

pub struct Tokenizer {
    pub vocab: Vec<String>,
    token_map: HashMap<String, u32>,
    unk_id: u32,
}

impl Tokenizer {
    pub fn from_vocab(vocab: Vec<String>) -> Self {
        let unk_id = vocab
            .iter()
            .position(|s| matches!(s.as_str(), "<unk>" | "[UNK]" | "<UNK>"))
            .unwrap_or(0) as u32;
        let token_map = vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        Self { vocab, token_map, unk_id }
    }

    /// Split by whitespace, then look up each word; unknown words hash into the
    /// vocabulary space so every token produces a meaningful node signal.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .map(|word| {
                self.token_map.get(word).copied().unwrap_or_else(|| {
                    let n = self.vocab.len().max(1) as u32;
                    djb2(word) % n
                })
            })
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| self.vocab.get(id as usize))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

fn djb2(s: &str) -> u32 {
    let mut h: u32 = 5381;
    for b in s.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u32);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_words_map_to_their_index() {
        let vocab = vec!["hello".into(), "world".into(), "<unk>".into()];
        let tok = Tokenizer::from_vocab(vocab);
        assert_eq!(tok.encode("hello world"), vec![0, 1]);
    }

    #[test]
    fn unknown_word_hashes_into_vocab_range() {
        let vocab: Vec<String> = (0..256).map(|i| format!("tok{i}")).collect();
        let tok = Tokenizer::from_vocab(vocab.clone());
        let ids = tok.encode("zzz_unknown");
        assert_eq!(ids.len(), 1);
        assert!((ids[0] as usize) < vocab.len());
    }

    #[test]
    fn decode_reverses_encode_for_known_words() {
        let vocab = vec!["a".into(), "b".into(), "c".into()];
        let tok = Tokenizer::from_vocab(vocab);
        let ids = tok.encode("a b c");
        assert_eq!(tok.decode(&ids), "a b c");
    }
}
