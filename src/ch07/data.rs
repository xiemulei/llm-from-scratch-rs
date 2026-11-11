//! 7.1 – 7.4：指令微调数据集。
//!
//! 和前面两章最大的不同：**损失只对回答部分计算**。
//!
//! ```text
//! [### Instruction:  把这句话改成被动语态  ### Input:  我把书读完了  ### Response:  书被我读完了]
//!  └─────────────── 不计算损失 ───────────────┘              └── 只在这里算 ──┘
//! ```
//!
//! 为什么？我们想让模型学会的是「该怎么回答」，而不是「复述指令」。
//! 如果指令部分也算损失，模型会花力气去预测用户给的输入，纯属浪费，
//! 而且会稀释真正有用的信号。
//!
//! 实现方式是给每个位置一个布尔掩码：指令区为 `false`，回答区为 `true`，
//! `masked_cross_entropy` 只对 `true` 的位置求平均。

use anyhow::Result;
use serde::Deserialize;

use crate::ch02::gpt2_tokenizer::Gpt2Tokenizer;

#[derive(Debug, Deserialize)]
struct RawEntry {
    instruction: String,
    #[serde(default)]
    input: String,
    output: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InstructionExample {
    pub instruction: String,
    pub input: String,
    pub output: String,
}

/// 读取书中提供的 instruction-data.json（1100 条）。
pub fn load_json(path: &str) -> Result<Vec<InstructionExample>> {
    let text = std::fs::read_to_string(path)?;
    let raw: Vec<RawEntry> = serde_json::from_str(&text)?;
    Ok(raw
        .into_iter()
        .map(|r| InstructionExample {
            instruction: r.instruction,
            input: r.input,
            output: r.output,
        })
        .collect())
}

/// 书中 7.2 的 Alpaca 风格模板（`input` 为空时省略 Input 段）。
pub fn format_input(e: &InstructionExample) -> String {
    if e.input.trim().is_empty() {
        format!(
            "Below is an instruction that describes a task. \
             Write a response that appropriately completes the request.\n\n\
             ### Instruction:\n{}\n\n### Response:\n",
            e.instruction.trim()
        )
    } else {
        format!(
            "Below is an instruction that describes a task, paired with an input \
             that provides further context. Write a response that appropriately \
             completes the request.\n\n\
             ### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n",
            e.instruction.trim(),
            e.input.trim()
        )
    }
}

/// 训练样本：完整词元序列 + 每个位置是否计入损失。
#[derive(Debug, Clone)]
pub struct EncodedInstruction {
    pub ids: Vec<u32>,
    /// `mask[i] == true` 表示第 i 个词元属于回答部分
    pub mask: Vec<bool>,
}

/// 编码一条样本。
///
/// `ids = encode(指令 + 回答)`，掩码在指令部分为 `false`、回答部分为 `true`。
/// 注意目标要整体左移一位（第 2 章就讲过的 next-token 错位），
/// 所以掩码也要跟着左移——这一步放在批处理里做。
pub fn encode(e: &InstructionExample, tk: &Gpt2Tokenizer) -> EncodedInstruction {
    let prompt = format_input(e);
    let prompt_ids = tk.encode(&prompt);
    let response_ids = tk.encode(&e.output);
    // 回答结束后补一个结束符，让模型学会「说完了」
    let end_id = tk.encode_with_special(&format!("{}<|endoftext|>", e.output));
    let response_ids = if end_id.len() > response_ids.len() {
        end_id
    } else {
        response_ids
    };

    let mut ids = prompt_ids.clone();
    ids.extend_from_slice(&response_ids);

    let mut mask = vec![false; prompt_ids.len()];
    mask.extend(std::iter::repeat(true).take(response_ids.len()));

    EncodedInstruction { ids, mask }
}

/// 一个批次的张量。
///
/// - `inputs`: `[batch, seq]`，右侧用 50256 补齐
/// - `targets`: `[batch, seq]`，为 `inputs` 左移一位（末尾补 0），**始终是合法词元 ID**
/// - `mask`: `[batch, seq]`，`true` 表示该位置计入损失
///
/// 为什么不把无效位置的 target 设成 -100（书里的 `ignore_index`）？
/// 因为 Burn 的 `gather` 不接受非法下标。我们把「忽略」这件事完全交给 mask，
/// 数据上更安全，语义也等价。
#[derive(Clone)]
pub struct InstructionBatch {
    pub inputs: burn::tensor::Tensor<2, burn::tensor::Int>,
    pub targets: burn::tensor::Tensor<2, burn::tensor::Int>,
    pub mask: burn::tensor::Tensor<2, burn::tensor::Bool>,
}

impl InstructionBatch {
    pub fn len(&self) -> usize {
        self.inputs.dims()[0]
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// 组批次：每个 batch 补齐到该 batch 内最长序列。
pub fn make_batches(
    data: &[EncodedInstruction],
    batch_size: usize,
    device: &burn::tensor::Device,
) -> Vec<InstructionBatch> {
    use burn::tensor::{Bool, Int, Tensor, TensorData};

    data.chunks(batch_size)
        .map(|chunk| {
            let max_len = chunk.iter().map(|e| e.ids.len()).max().unwrap_or(1).max(2);

            let mut inputs = Vec::with_capacity(chunk.len() * max_len);
            let mut targets = Vec::with_capacity(chunk.len() * max_len);
            let mut mask = Vec::with_capacity(chunk.len() * max_len);

            for e in chunk {
                let len = e.ids.len();
                for i in 0..max_len {
                    if i < len {
                        inputs.push(e.ids[i] as i64);
                        // 目标 = 下一个词元；掩码也跟着左移一位
                        targets.push(*e.ids.get(i + 1).unwrap_or(&0) as i64);
                        mask.push(*e.mask.get(i + 1).unwrap_or(&false));
                    } else {
                        inputs.push(50256); // 填充词元
                        targets.push(0);
                        mask.push(false);
                    }
                }
            }

            InstructionBatch {
                inputs: Tensor::<2, Int>::from_data(
                    TensorData::new(inputs, [chunk.len(), max_len]),
                    device,
                ),
                targets: Tensor::<2, Int>::from_data(
                    TensorData::new(targets, [chunk.len(), max_len]),
                    device,
                ),
                mask: Tensor::<2, Bool>::from_data(
                    TensorData::new(mask, [chunk.len(), max_len]),
                    device,
                ),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example(input: &str) -> InstructionExample {
        InstructionExample {
            instruction: "Edit the sentence.".into(),
            input: input.into(),
            output: "Fixed.".into(),
        }
    }

    #[test]
    fn template_omits_input_when_empty() {
        let p = format_input(&example(""));
        assert!(p.contains("### Instruction:"));
        assert!(!p.contains("### Input:"));
        assert!(p.ends_with("### Response:\n"));
    }

    #[test]
    fn template_includes_input_when_present() {
        let p = format_input(&example("bad sentence"));
        assert!(p.contains("### Input:\nbad sentence"));
    }

    #[test]
    fn mask_covers_only_the_response() {
        let tk = Gpt2Tokenizer::new().unwrap();
        let e = example("");
        let enc = encode(&e, &tk);

        assert_eq!(enc.ids.len(), enc.mask.len());
        // 掩码必须是「全 false 前缀 + 全 true 后缀」
        let first_true = enc.mask.iter().position(|&b| b).unwrap();
        assert!(enc.mask[..first_true].iter().all(|&b| !b));
        assert!(enc.mask[first_true..].iter().all(|&b| b));
        // 回答部分应该至少有 2 个词元
        assert!(enc.mask.len() - first_true >= 2);
    }

    #[test]
    fn decode_of_response_part_matches_output() {
        let tk = Gpt2Tokenizer::new().unwrap();
        let e = example("");
        let enc = encode(&e, &tk);
        let first_true = enc.mask.iter().position(|&b| b).unwrap();
        let decoded = tk.decode(&enc.ids[first_true..]);
        assert!(decoded.starts_with("Fixed."), "实际: {decoded:?}");
    }

    #[test]
    fn batches_shift_targets_by_one() {
        let device = burn::tensor::Device::default();
        let data = vec![EncodedInstruction {
            ids: vec![10, 11, 12, 13],
            mask: vec![false, false, true, true],
        }];
        let b = &make_batches(&data, 4, &device)[0];
        assert_eq!(b.inputs.dims(), [1, 4]);

        let inputs = b.inputs.to_data().try_to_vec::<i32>().unwrap();
        let targets = b.targets.to_data().try_to_vec::<i32>().unwrap();
        // target[i] = input[i+1]
        assert_eq!(inputs, vec![10, 11, 12, 13]);
        assert_eq!(targets, vec![11, 12, 13, 0]);
    }

    #[test]
    fn padding_is_masked_out() {
        let device = burn::tensor::Device::default();
        let data = vec![
            EncodedInstruction { ids: vec![1, 2, 3, 4], mask: vec![false, false, true, true] },
            EncodedInstruction { ids: vec![1, 2], mask: vec![false, true] },
        ];
        let b = &make_batches(&data, 2, &device)[0];
        assert_eq!(b.inputs.dims(), [2, 4]);
        // ⚠️ Bool 张量的存储布局是后端相关的：flex 用 Bool(Native)，
        // wgpu 用 Bool(U32)。直接 try_to_vec::<bool>() 在 GPU 上会 DTypeMismatch，
        // 所以先转成整数再读回，这样两个后端都通用。
        let mask: Vec<i32> = b
            .mask
            .clone()
            .int()
            .to_data()
            .try_to_vec::<i32>()
            .unwrap();
        let mask: Vec<bool> = mask.into_iter().map(|v| v != 0).collect();
        // 第二条样本只有 2 个词元：[1, 2]，掩码 [false, true]。
        // 左移一位后第 0 个位置的目标是词元 2（属于回答）→ true；
        // 第 1、2、3 列是填充或越界 → 必须被掩掉。
        assert_eq!(mask[4..], [true, false, false, false]);
    }
}
