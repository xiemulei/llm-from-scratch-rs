//! 5.1：语料 → 词元 → 训练批次。
//!
//! 书中用的是 Edith Wharton 的短篇《The Verdict》（公有领域），这里换成自写文本，
//! 流程完全一致：
//!
//! ```text
//! 原始文本 ──GPT-2 分词器──> Vec<u32> ──滑动窗口──> (input, target) 批次
//! ```
//!
//! 注意预训练不需要任何标注：**文本自己就是标签**。这也是"无监督"的含义——
//! 准确说是"自监督"（self-supervised）。

use anyhow::Result;
use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::ch02::dataset::GptDataset;
use crate::ch02::gpt2_tokenizer::Gpt2Tokenizer;

/// 内置语料：第 2 章那段自写英文，重复若干次凑够训练样本。
pub fn builtin_corpus(repeat: usize) -> String {
    crate::ch02::SAMPLE_TEXT.repeat(repeat)
}

/// 从文件读取语料；文件不存在则回退到内置语料。
pub fn load_corpus(path: &str, repeat: usize) -> Result<String> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(text),
        Err(_) => {
            println!("（找不到 {path}，使用内置语料）");
            Ok(builtin_corpus(repeat))
        }
    }
}

/// 把文本变成词元 ID 序列，末尾补一个 `<|endoftext|>` 表示文档结束。
pub fn encode(text: &str, tk: &Gpt2Tokenizer) -> Vec<u32> {
    tk.encode_with_special(&format!("{text}<|endoftext|>"))
}

/// 把词元序列切成训练批次。
///
/// 返回的每个元组是 `(inputs [b, t], targets [b, t])`，已经是可以直接喂模型的张量。
pub fn make_batches(
    tokens: &[u32],
    max_length: usize,
    stride: usize,
    batch_size: usize,
    device: &Device,
) -> Vec<(Tensor<2, Int>, Tensor<2, Int>)> {
    let ds = GptDataset::new(tokens.to_vec(), max_length, stride);
    let batches = crate::ch02::dataset::make_batches(&ds, batch_size, true, true);

    batches
        .into_iter()
        .map(|b| {
            let (inputs, targets, shape) = b.to_flat();
            (
                Tensor::<2, Int>::from_data(TensorData::new(inputs, shape), device),
                Tensor::<2, Int>::from_data(TensorData::new(targets, shape), device),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_have_expected_shape() {
        let device = Device::default();
        let tokens: Vec<u32> = (0..200).map(|i| (i % 100) as u32).collect();
        let batches = make_batches(&tokens, 16, 16, 4, &device);
        assert!(!batches.is_empty());
        for (x, y) in &batches {
            assert_eq!(x.dims(), [4, 16]);
            assert_eq!(y.dims(), [4, 16]);
        }
    }
}
