//! 7.5 – 7.6：带掩码损失的训练。
//!
//! 指令微调本身和预训练是同一件事（都是 next-token 预测），只有一处不同：
//! 损失只在回答部分计算。其余流程——预训练权重、交叉熵、AdamW——完全一样。
//!
//! 书里用 gpt2-medium（355M）。我们这台机器没有 CUDA，355M 在 CPU 上跑不动，
//! 所以这里用 gpt2-small（124M），流程完全一致。

use burn::tensor::activation::log_softmax;
use burn::tensor::{Bool, Device, Int, Tensor};

use crate::ch04::gpt::Gpt;
use crate::ch07::data::InstructionBatch;

/// 带掩码的交叉熵。
///
/// `logits [batch, seq, vocab]`，`targets [batch, seq]`，`mask [batch, seq]`。
/// 只对 `mask == true` 的位置求平均。
pub fn masked_cross_entropy(
    logits: Tensor<3>,
    targets: Tensor<2, Int>,
    mask: Tensor<2, Bool>,
) -> Tensor<1> {
    let [batch, seq, vocab] = logits.dims();
    let n = batch * seq;

    // 展平后逐位置算 -log p(target)
    let log_probs = log_softmax(logits.reshape([n, vocab]), 1);
    let picked = log_probs.gather(1, targets.reshape([n, 1])); // [n, 1]
    let nll = picked.reshape([n]).neg(); // [n]

    let mask = mask.reshape([n]);
    let kept = nll.mask_fill(mask.clone().bool_not(), 0.0);
    let count = mask.clone().float().sum().clamp_min(1.0);

    kept.sum() / count
}

/// 训练配置（书中 7.6：5 个 epoch、lr 5e-5、weight decay 0.1）。
pub struct InstructConfig {
    pub epochs: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub eval_interval: usize,
}

impl Default for InstructConfig {
    fn default() -> Self {
        Self {
            epochs: 1,
            lr: 5e-5,
            weight_decay: 0.1,
            eval_interval: 20,
        }
    }
}

pub fn evaluate(model: &Gpt, batches: &[InstructionBatch]) -> f32 {
    if batches.is_empty() {
        return f32::NAN;
    }
    let mut total = 0.0;
    for b in batches {
        let logits = model.forward(b.inputs.clone());
        total += masked_cross_entropy(logits, b.targets.clone(), b.mask.clone())
            .into_scalar::<f32>();
    }
    total / batches.len() as f32
}

pub fn fine_tune(
    mut model: Gpt,
    train: &[InstructionBatch],
    val: &[InstructionBatch],
    cfg: &InstructConfig,
) -> (Gpt, Vec<(usize, f32, f32)>) {
    use burn::optim::{AdamWConfig, GradientsParams};

    let mut optim = AdamWConfig::new()
        .with_weight_decay(cfg.weight_decay as f32)
        .init();

    let mut history = Vec::new();
    let total_steps = cfg.epochs * train.len();
    let mut step = 0usize;
    let mut acc = 0.0f32;
    let mut acc_n = 0usize;

    println!(
        "开始指令微调：{} epoch × {} 步 = {} 步",
        cfg.epochs,
        train.len(),
        total_steps
    );

    for epoch in 1..=cfg.epochs {
        for b in train {
            let logits = model.forward(b.inputs.clone());
            let loss = masked_cross_entropy(logits, b.targets.clone(), b.mask.clone());
            acc += loss.clone().into_scalar::<f32>();
            acc_n += 1;

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(cfg.lr, model, grads);

            step += 1;
            if step % cfg.eval_interval == 0 || step == total_steps {
                let train_loss = acc / acc_n as f32;
                let val_loss = evaluate(&model, val);
                println!(
                    "  epoch {epoch} step {step:>4}/{total_steps:<4} train {train_loss:.4} val {val_loss:.4}"
                );
                history.push((step, train_loss, val_loss));
                acc = 0.0;
                acc_n = 0;
            }
        }
    }
    (model, history)
}

/// 推理：给定指令（和可选的输入），自回归生成回答。
pub fn generate_response(
    model: &Gpt,
    example: &crate::ch07::data::InstructionExample,
    tk: &crate::ch02::gpt2_tokenizer::Gpt2Tokenizer,
    max_new_tokens: usize,
    device: &Device,
) -> String {
    use crate::ch04::generate::{generate, Sampling};

    let prompt = crate::ch07::data::format_input(example);
    let ids: Vec<i64> = tk.encode(&prompt).iter().map(|&x| x as i64).collect();
    let n = ids.len();

    let idx = Tensor::<2, Int>::from_data(
        burn::tensor::TensorData::new(ids, [1, n]),
        device,
    );

    // 温度调低一点，指令微调后的模型需要更"守规矩"
    let out = generate(
        model,
        idx,
        max_new_tokens,
        Sampling::TopK {
            k: 25,
            temperature: 0.7,
        },
        device,
    );

    let all: Vec<u32> = out
        .to_data()
        .try_to_vec::<i32>()
        .unwrap()
        .into_iter()
        .map(|x| x as u32)
        .collect();

    let text = tk.decode(&all[n..]);
    // 只取到第一个结束符为止
    text.split("<|endoftext|>").next().unwrap_or("").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    fn device() -> Device {
        Device::default()
    }

    #[test]
    fn mask_ignores_instruction_positions() {
        let d = device();
        // 两个位置：位置 0 掩掉，位置 1 保留
        let logits = Tensor::<3>::from_data(
            TensorData::new(vec![0.0f32, 10.0, 10.0, 0.0], [1, 2, 2]),
            &d,
        );
        // 目标：位置 0 预测错（真实是 1，模型给 0 高分），位置 1 预测对
        let targets = Tensor::<2, Int>::from_data(TensorData::new(vec![1i64, 0], [1, 2]), &d);
        let mask = Tensor::<2, Bool>::from_data(TensorData::new(vec![false, true], [1, 2]), &d);

        let loss = masked_cross_entropy(logits, targets, mask).into_scalar::<f32>();
        // 只算位置 1（完美预测），损失应≈0；若把位置 0 也算进去会很大
        assert!(loss < 1e-3, "被掩掉的位置不应影响损失，实际 {loss}");
    }

    #[test]
    fn all_masked_does_not_divide_by_zero() {
        let d = device();
        let logits = Tensor::<3>::zeros([1, 2, 3], &d);
        let targets = Tensor::<2, Int>::zeros([1, 2], &d);
        let mask = Tensor::<2, Bool>::from_data(TensorData::new(vec![false, false], [1, 2]), &d);
        let loss = masked_cross_entropy(logits, targets, mask).into_scalar::<f32>();
        assert!(loss.is_finite(), "全掩码时损失应为 0 而不是 NaN");
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn full_mask_equals_plain_cross_entropy() {
        let d = device();
        let logits = Tensor::<3>::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 0.5, 0.1, 0.9], [1, 2, 3]),
            &d,
        );
        let targets = Tensor::<2, Int>::from_data(TensorData::new(vec![2i64, 0], [1, 2]), &d);
        let mask = Tensor::<2, Bool>::from_data(TensorData::new(vec![true, true], [1, 2]), &d);

        let masked = masked_cross_entropy(logits.clone(), targets.clone(), mask)
            .into_scalar::<f32>();
        let plain = crate::ch05::loss::cross_entropy(logits, targets).into_scalar::<f32>();
        assert!((masked - plain).abs() < 1e-5);
    }
}
