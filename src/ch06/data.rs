//! 6.1 – 6.3：垃圾短信数据集。
//!
//! 数据来自 UCI 的 SMS Spam Collection：5574 条短信，4827 条正常（ham）+ 747 条垃圾（spam）。
//!
//! 处理流程和书中一致：
//!
//! 1. 打乱（固定种子，保证可复现）
//! 2. 按 70% / 10% / 20% 切成 训练集 / 验证集 / 测试集
//! 3. **测试集下采样成平衡集**（747 spam + 747 ham）——否则 86% 都是 ham，
//!    模型只要全猜 ham 就有 86% 准确率，指标毫无意义
//! 4. 每条短信编码成词元 ID，同一个 batch 内右侧补齐到最长
//!
//! 关于填充：GPT-2 没有专门的 pad 词元，书中直接用 `<|endoftext|>`（50256）。
//! 因为我们只取**最后一个真实词元**的隐藏状态做分类，右侧填充不会影响结果。

use anyhow::{bail, Result};
use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::ch02::gpt2_tokenizer::Gpt2Tokenizer;

/// `<|endoftext|>`，兼作填充词元
pub const PAD_TOKEN: u32 = 50256;

#[derive(Debug, Clone, PartialEq)]
pub struct Example {
    /// 0 = ham（正常），1 = spam（垃圾）
    pub label: usize,
    pub text: String,
}

/// 读取 TSV：第一行是表头 `Label\tText`，之后每行 `ham|spam \t 正文`。
pub fn load_tsv(path: &str) -> Result<Vec<Example>> {
    let content = std::fs::read_to_string(path)?;
    let mut out = Vec::new();

    for (i, line) in content.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let Some((label, text)) = line.split_once('\t') else {
            continue;
        };
        let label = match label.trim().to_ascii_lowercase().as_str() {
            "ham" => 0,
            "spam" => 1,
            "label" => continue, // 表头
            other => bail!("第 {i} 行标签无法识别: {other}"),
        };
        out.push(Example {
            label,
            text: text.to_string(),
        });
    }
    Ok(out)
}

/// 确定性洗牌（线性同余），和 Python 的 `random.seed(123)` 起同样作用：可复现。
pub fn shuffle(examples: &mut [Example], seed: u64) {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    for i in (1..examples.len()).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        examples.swap(i, j);
    }
}

/// 70 / 10 / 20 切分。
pub fn split(examples: Vec<Example>) -> (Vec<Example>, Vec<Example>, Vec<Example>) {
    let n = examples.len();
    let train_end = n * 70 / 100;
    let val_end = train_end + n * 10 / 100;

    let mut v = examples;
    let test = v.split_off(val_end);
    let val = v.split_off(train_end);
    (v, val, test)
}

/// 下采样成平衡集：取数量较少的那一类的最小数量。
pub fn balance(examples: Vec<Example>) -> Vec<Example> {
    let spam: Vec<Example> = examples.iter().filter(|e| e.label == 1).cloned().collect();
    let ham: Vec<Example> = examples.iter().filter(|e| e.label == 0).cloned().collect();
    let n = spam.len().min(ham.len());

    let mut out = Vec::with_capacity(2 * n);
    out.extend(spam.into_iter().take(n));
    out.extend(ham.into_iter().take(n));
    out
}

/// 一条编码后的样本
#[derive(Debug, Clone)]
pub struct Encoded {
    pub ids: Vec<u32>,
    pub label: usize,
}

pub fn encode(examples: &[Example], tk: &Gpt2Tokenizer) -> Vec<Encoded> {
    examples
        .iter()
        .map(|e| Encoded {
            ids: tk.encode(&e.text),
            label: e.label,
        })
        .collect()
}

/// 一个批次的输入张量。
///
/// `inputs: [batch, max_len]`（右侧用 PAD_TOKEN 补齐）
/// `targets: [batch]`
#[derive(Debug, Clone)]
pub struct Batch {
    pub inputs: Tensor<2, Int>,
    pub targets: Tensor<1, Int>,
}

impl Batch {
    pub fn len(&self) -> usize {
        self.inputs.dims()[0]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// 把样本切成批次，并在每个批次内补齐到该批次的最长序列。
pub fn make_batches(
    data: &[Encoded],
    batch_size: usize,
    device: &Device,
) -> Vec<Batch> {
    data.chunks(batch_size)
        .map(|chunk| {
            let max_len = chunk.iter().map(|e| e.ids.len()).max().unwrap_or(1).max(1);

            let mut inputs = Vec::with_capacity(chunk.len() * max_len);
            let mut targets = Vec::with_capacity(chunk.len());
            for e in chunk {
                inputs.extend(e.ids.iter().map(|&x| x as i64));
                // 右侧补齐
                inputs.extend(std::iter::repeat(PAD_TOKEN as i64).take(max_len - e.ids.len()));
                targets.push(e.label as i64);
            }

            Batch {
                inputs: Tensor::<2, Int>::from_data(
                    TensorData::new(inputs, [chunk.len(), max_len]),
                    device,
                ),
                targets: Tensor::<1, Int>::from_data(
                    TensorData::new(targets, [chunk.len()]),
                    device,
                ),
            }
        })
        .collect()
}

/// 统计标签分布，用于打印数据集概况。
pub fn label_counts(data: &[Example]) -> (usize, usize) {
    let spam = data.iter().filter(|e| e.label == 1).count();
    (data.len() - spam, spam)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<Example> {
        vec![
            Example { label: 0, text: "hello".into() },
            Example { label: 1, text: "win money".into() },
            Example { label: 1, text: "free prize".into() },
            Example { label: 0, text: "see you".into() },
            Example { label: 0, text: "lunch?".into() },
        ]
    }

    #[test]
    fn split_proportions_are_70_10_20() {
        let data: Vec<Example> = (0..100)
            .map(|i| Example { label: i % 2, text: format!("t{i}") })
            .collect();
        let (train, val, test) = split(data);
        assert_eq!((train.len(), val.len(), test.len()), (70, 10, 20));
    }

    #[test]
    fn balance_equalizes_classes() {
        let balanced = balance(sample());
        let (ham, spam) = label_counts(&balanced);
        assert_eq!(ham, spam);
        assert_eq!(balanced.len(), 4);
    }

    #[test]
    fn shuffle_is_deterministic() {
        let mut a = sample();
        let mut b = sample();
        shuffle(&mut a, 123);
        shuffle(&mut b, 123);
        assert_eq!(a, b);
    }

    #[test]
    fn batches_are_padded_to_batch_max() {
        let device = Device::default();
        let data = vec![
            Encoded { ids: vec![1, 2, 3], label: 0 },
            Encoded { ids: vec![4], label: 1 },
            Encoded { ids: vec![5, 6], label: 0 },
        ];
        let batches = make_batches(&data, 8, &device);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].inputs.dims(), [3, 3]);

        let inputs = batches[0].inputs.to_data().try_to_vec::<i32>().unwrap();
        assert_eq!(inputs, vec![1, 2, 3, 4, 50256, 50256, 5, 6, 50256]);
        let targets = batches[0].targets.to_data().try_to_vec::<i32>().unwrap();
        assert_eq!(targets, vec![0, 1, 0]);
    }

    #[test]
    fn batch_size_controls_chunking() {
        let device = Device::default();
        let data: Vec<Encoded> = (0..10)
            .map(|i| Encoded { ids: vec![i as u32], label: 0 })
            .collect();
        let batches = make_batches(&data, 4, &device);
        assert_eq!(batches.len(), 3);
        assert_eq!(batches.last().unwrap().len(), 2);
    }
}
