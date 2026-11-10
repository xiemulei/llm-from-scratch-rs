//! 2.2 – 2.4：把文本切成词元，并映射成整数 ID。
//!
//! 原书先用最朴素的方式建立直觉：按空白和标点切词 → 建词表 → 编码 / 解码，
//! 然后指出它的两个致命缺陷（词表爆炸、无法处理未登录词），从而引出 2.5 的 BPE。
//! 这一段我们照做一遍。

use std::collections::HashMap;

/// 未登录词的特殊词元，对应书中的 `<|unk|>`。
pub const UNK: &str = "<|unk|>";
/// 文本结束符，GPT-2 用它区分两篇独立文档，对应书中的 `<|endoftext|>`。
pub const END_OF_TEXT: &str = "<|endoftext|>";

/// 会被单独切出来的标点，取自书中的正则 `[,.:;?_!"()\\']|--`。
const PUNCTUATION: &[char] = &[
    ',', '.', ':', ';', '?', '_', '!', '"', '\'', '(', ')', '-',
];

/// 2.2 的朴素分词：按空白切分，并把标点单独剥离。
///
/// 对比原文的正则版本，这里手写循环是为了不引入 `regex` 依赖，
/// 行为等价（`--` 会被拆成两个 `-`，不影响后续演示）。
pub fn simple_tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for word in text.split_whitespace() {
        let mut current = String::new();
        for ch in word.chars() {
            if PUNCTUATION.contains(&ch) {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                tokens.push(ch.to_string());
            } else {
                current.push(ch);
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }
    }
    tokens
}

/// 2.3 / 2.4 的简易词表：字符串 ↔ 整数 ID 的双向映射。
///
/// 这是所有分词器的本质——一张查找表。BPE、SentencePiece、tiktoken 都是在这张表上做文章。
#[derive(Debug, Clone, Default)]
pub struct SimpleVocab {
    str_to_int: HashMap<String, usize>,
    int_to_str: HashMap<usize, String>,
}

impl SimpleVocab {
    /// 从语料构建词表。`specials` 是预留的特殊词元，会被排在最前面。
    pub fn from_corpus(corpus: &str, specials: &[&str]) -> Self {
        let mut vocab = SimpleVocab::default();
        for s in specials {
            vocab.insert(s);
        }
        for token in simple_tokenize(corpus) {
            vocab.insert(&token);
        }
        vocab
    }

    fn insert(&mut self, token: &str) {
        if self.str_to_int.contains_key(token) {
            return;
        }
        let id = self.str_to_int.len();
        self.str_to_int.insert(token.to_string(), id);
        self.int_to_str.insert(id, token.to_string());
    }

    pub fn len(&self) -> usize {
        self.str_to_int.len()
    }

    pub fn is_empty(&self) -> bool {
        self.str_to_int.is_empty()
    }

    /// 编码。未登录词统一落到 `<|unk|>`。
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let unk = self.str_to_int[UNK];
        simple_tokenize(text)
            .iter()
            .map(|t| self.str_to_int.get(t).copied().unwrap_or(unk))
            .collect()
    }

    /// 解码。间隔规则与书中一致：标点前不加空格。
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut text = String::new();
        for &id in ids {
            let token = &self.int_to_str[&id];
            let is_punct = token.chars().count() == 1
                && PUNCTUATION.contains(&token.chars().next().unwrap());
            if !text.is_empty() && !is_punct {
                text.push(' ');
            }
            text.push_str(token);
        }
        text
    }

    pub fn id_of(&self, token: &str) -> Option<usize> {
        self.str_to_int.get(token).copied()
    }

    pub fn token_of(&self, id: usize) -> Option<&str> {
        self.int_to_str.get(&id).map(|s| s.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CORPUS: &str = "Hello, do you like tea? In the sunlit terraces of the palace.";

    #[test]
    fn tokenizer_splits_punctuation() {
        assert_eq!(
            simple_tokenize("Hello, world!"),
            vec!["Hello", ",", "world", "!"]
        );
    }

    #[test]
    fn roundtrip_for_known_text() {
        let vocab = SimpleVocab::from_corpus(CORPUS, &[UNK, END_OF_TEXT]);
        let ids = vocab.encode("Hello, do you like tea?");
        assert_eq!(vocab.decode(&ids), "Hello, do you like tea?");
    }

    #[test]
    fn unknown_words_map_to_unk() {
        let vocab = SimpleVocab::from_corpus(CORPUS, &[UNK, END_OF_TEXT]);
        let ids = vocab.encode("Hello, unknownword");
        let unk = vocab.id_of(UNK).unwrap();
        assert_eq!(*ids.last().unwrap(), unk);
    }
}
