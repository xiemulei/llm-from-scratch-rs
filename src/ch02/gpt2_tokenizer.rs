//! 2.5（续）：GPT-2 官方字节对编码。
//!
//! 上一节我们手写了一个可训练的 BPE，那理解原理足够，但不能直接用——
//! GPT-2 的预训练权重是按它自己的词表训练的，词表对不上，权重就全废了。
//!
//! 官方词表（50257 个词元）由 `tiktoken-rs` 提供，它会在首次运行时下载
//! GPT-2 的 `vocab.bpe` / `encoder.json` 并缓存。

use anyhow::{Context, Result};
use tiktoken_rs::CoreBPE;

/// GPT-2 词表大小：50000 个 BPE 词元 + 256 个字节 + 1 个 `<|endoftext|>`
pub const GPT2_VOCAB_SIZE: usize = 50257;
/// `<|endoftext|>` 的词元 ID
pub const END_OF_TEXT_ID: u32 = 50256;

pub struct Gpt2Tokenizer {
    bpe: &'static CoreBPE,
}

impl Gpt2Tokenizer {
    /// 加载 GPT-2 分词器（首次运行需要联网下载词表）。
    pub fn new() -> Result<Self> {
        let bpe = tiktoken_rs::bpe_for_model("gpt2")
            .context("无法加载 GPT-2 词表（检查网络或代理）")?;
        Ok(Self { bpe })
    }

    /// 普通编码：不识别特殊词元，`<|endoftext|>` 会被当普通文本切开。
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.bpe.encode_ordinary(text)
    }

    /// 带特殊词元的编码：`<|endoftext|>` 会被识别为 ID 50256。
    ///
    /// 书中把它插在两段独立文本之间，让模型知道"这里是一篇新文档"。
    pub fn encode_with_special(&self, text: &str) -> Vec<u32> {
        self.bpe.encode_with_special_tokens(text)
    }

    /// 解码。词元边界处可能出现半个 UTF-8 字符，这里做有损转换。
    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes = self.bpe.decode_bytes(ids).unwrap_or_default();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// 把文本切成词元字符串，用于观察 BPE 实际学到了什么。
    ///
    /// 例：`"Hello, world!"` → `["Hello", ",", " world", "!"]`
    /// 注意 ` world` 前面的空格——GPT-2 里空格属于词元的一部分。
    pub fn tokenize_to_strings(&self, text: &str) -> Vec<String> {
        self.encode(text)
            .iter()
            .map(|&id| {
                let bytes = self.bpe.decode_bytes(&[id]).unwrap_or_default();
                String::from_utf8_lossy(&bytes).into_owned()
            })
            .collect()
    }

    /// 带 ID 的词元列表，对应书中 `list(zip(tokens, token_ids))` 的输出。
    pub fn tokenize_with_ids(&self, text: &str) -> Vec<(String, u32)> {
        self.encode(text)
            .into_iter()
            .map(|id| {
                let bytes = self.bpe.decode_bytes(&[id]).unwrap_or_default();
                (String::from_utf8_lossy(&bytes).into_owned(), id)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 集成测试：依赖网络下载词表，用 `cargo test -- --ignored` 运行
    #[test]
    #[ignore = "需要联网下载 GPT-2 词表"]
    fn known_ids_match_the_book() {
        let tk = Gpt2Tokenizer::new().unwrap();
        // 书中 2.5 的输出：Hello, do you like tea? → 15496 11 466 345 588 8887 30
        assert_eq!(
            tk.encode("Hello, do you like tea?"),
            vec![15496, 11, 466, 345, 588, 8887, 30]
        );
        assert_eq!(tk.decode(&[15496, 11]), "Hello,");
    }

    #[test]
    #[ignore = "需要联网下载 GPT-2 词表"]
    fn endoftext_is_a_single_token() {
        let tk = Gpt2Tokenizer::new().unwrap();
        assert_eq!(tk.encode_with_special("<|endoftext|>"), vec![END_OF_TEXT_ID]);
        // 普通编码下它会被拆成 7 个词元
        assert!(tk.encode("<|endoftext|>").len() > 1);
    }

    #[test]
    #[ignore = "需要联网下载 GPT-2 词表"]
    fn whitespace_belongs_to_the_token() {
        let tk = Gpt2Tokenizer::new().unwrap();
        assert_eq!(tk.tokenize_to_strings("Hello, world!"), vec!["Hello", ",", " world", "!"]);
    }
}
