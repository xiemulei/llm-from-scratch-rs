//! 5.2 – 5.4：训练循环。
//!
//! ## 5.3 一个训练步
//!
//! ```text
//! logits = model(x)                   前向
//! loss   = cross_entropy(logits, y)   损失
//! grads  = loss.backward()            反向
//! model  = optimizer.step(lr, ...)    更新参数
//! ```
//!
//! ## 学习率调度
//!
//! 书里用「线性预热 + 余弦退火」：
//!
//! - **预热**（前 N 步）：学习率从 0 线性升到最大值。Adam 的动量估计在最初几步还不可靠，
//!   直接用大学习率容易把参数炸飞。
//! - **余弦退火**：`lr = min_lr + 0.5(1 + cos(π·progress))·(max_lr - min_lr)`，
//!   平滑地降到 `min_lr`。
//!
//! ## 梯度裁剪
//!
//! 梯度范数超过阈值就整体缩放。预训练时偶尔会遇到 loss 突然飙升（梯度爆炸），
//! 裁剪是最省事的保险。

use burn::grad_clipping::GradientClippingConfig;
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams};
use burn::tensor::{Int, Tensor};

use crate::ch04::gpt::Gpt;
use crate::ch05::loss::cross_entropy;

pub struct TrainConfig {
    pub epochs: usize,
    pub lr: f64,
    pub min_lr: f64,
    /// 预热步数
    pub warmup_steps: usize,
    pub weight_decay: f64,
    /// 梯度范数裁剪阈值，None 表示不裁剪
    pub grad_clip: Option<f64>,
    /// 每隔多少步在验证集上评估一次
    pub eval_interval: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        // 和书中 5.2 节一致：lr 4e-4、weight decay 0.1、warmup 10、clip 1.0
        Self {
            epochs: 10,
            lr: 4e-4,
            min_lr: 1e-5,
            warmup_steps: 10,
            weight_decay: 0.1,
            grad_clip: Some(1.0),
            eval_interval: 5,
        }
    }
}

#[derive(Debug, Default)]
pub struct TrainHistory {
    /// (全局步数, 训练损失)
    pub train_loss: Vec<(usize, f32)>,
    /// (全局步数, 验证损失)
    pub val_loss: Vec<(usize, f32)>,
}

impl TrainHistory {
    pub fn final_train_loss(&self) -> f32 {
        self.train_loss.last().map(|(_, l)| *l).unwrap_or(f32::NAN)
    }
}

/// 当前步的学习率（线性预热 + 余弦退火）。
pub fn lr_at(step: usize, total_steps: usize, cfg: &TrainConfig) -> f64 {
    if step < cfg.warmup_steps {
        // 预热：从 lr/10 线性升到 lr（不能从 0 开始，否则前几步等于没学）
        let t = (step + 1) as f64 / cfg.warmup_steps as f64;
        return cfg.lr * t;
    }
    let progress = (step - cfg.warmup_steps) as f64
        / (total_steps.saturating_sub(cfg.warmup_steps).max(1)) as f64;
    let progress = progress.clamp(0.0, 1.0);
    cfg.min_lr + 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()) * (cfg.lr - cfg.min_lr)
}

type Batch = (Tensor<2, Int>, Tensor<2, Int>);

/// 在给定批次上算平均损失（不反向传播）。
pub fn evaluate(model: &Gpt, batches: &[Batch]) -> f32 {
    if batches.is_empty() {
        return f32::NAN;
    }
    let mut total = 0.0f32;
    for (x, y) in batches {
        let logits = model.forward(x.clone());
        total += cross_entropy(logits, y.clone()).into_scalar::<f32>();
    }
    total / batches.len() as f32
}

/// 训练主循环。
///
/// ⚠️ `model` 必须建在 `device.clone().autodiff()` 上，批次张量也要在同一个设备上，
/// 否则 `backward()` 拿不到计算图。调用方负责这一点。
pub fn train(
    mut model: Gpt,
    train_batches: &[Batch],
    val_batches: &[Batch],
    cfg: &TrainConfig,
) -> (Gpt, TrainHistory) {
    let mut optim_cfg = AdamWConfig::new().with_weight_decay(cfg.weight_decay as f32);
    if let Some(clip) = cfg.grad_clip {
        optim_cfg = optim_cfg.with_grad_clipping(Some(GradientClippingConfig::Norm(clip as f32)));
    }
    let mut optim = optim_cfg.init();

    let steps_per_epoch = train_batches.len().max(1);
    let total_steps = cfg.epochs * steps_per_epoch;
    let mut history = TrainHistory::default();
    let mut global_step = 0usize;
    let mut running_loss = 0.0f32;
    let mut running_count = 0usize;

    println!(
        "开始训练：{} 个 epoch × {} 步 = {} 步，参数量 {}",
        cfg.epochs,
        steps_per_epoch,
        total_steps,
        model.num_params()
    );

    for epoch in 1..=cfg.epochs {
        for (x, y) in train_batches {
            let lr = lr_at(global_step, total_steps, cfg);

            let logits = model.forward(x.clone());
            let loss = cross_entropy(logits, y.clone());
            let loss_val = loss.clone().into_scalar::<f32>();

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(lr, model, grads);

            running_loss += loss_val;
            running_count += 1;
            global_step += 1;

            if global_step % cfg.eval_interval == 0 || global_step == total_steps {
                let avg = running_loss / running_count as f32;
                let val = evaluate(&model, val_batches);
                history.train_loss.push((global_step, avg));
                history.val_loss.push((global_step, val));
                println!(
                    "  epoch {epoch:>2} step {global_step:>4}/{total_steps:<4} lr {lr:.2e} train {avg:.4} val {val:.4}"
                );
                running_loss = 0.0;
                running_count = 0;
            }
        }
    }

    (model, history)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ch04::gpt::GPTConfig;
    use burn::tensor::{Device, TensorData};

    #[test]
    fn warmup_is_monotonic_and_peaks_at_lr() {
        let cfg = TrainConfig {
            warmup_steps: 10,
            lr: 4e-4,
            min_lr: 1e-5,
            ..Default::default()
        };
        let mut prev = 0.0;
        for step in 0..10 {
            let lr = lr_at(step, 100, &cfg);
            assert!(lr > prev, "预热阶段学习率必须单调上升");
            prev = lr;
        }
        assert!((prev - cfg.lr).abs() < 1e-12, "预热结束应正好达到最大学习率");
    }

    #[test]
    fn cosine_decays_to_min_lr() {
        let cfg = TrainConfig {
            warmup_steps: 10,
            lr: 4e-4,
            min_lr: 1e-5,
            ..Default::default()
        };
        let at_end = lr_at(110, 110, &cfg);
        assert!((at_end - cfg.min_lr).abs() < 1e-9, "末期应降到 min_lr");
        let middle = lr_at(60, 110, &cfg);
        assert!(middle < cfg.lr && middle > cfg.min_lr);
    }

    #[test]
    fn one_step_reduces_loss_on_a_tiny_problem() {
        // 单层小模型学一个常量映射，几步之内损失必须下降
        let device = Device::default().autodiff();
        let cfg = GPTConfig::new(8, 8, 8, 2, 1).with_drop_rate(0.0);
        let model = Gpt::new(&cfg, &device);

        let x = Tensor::<2, Int>::from_data(TensorData::new(vec![0i64; 16], [2, 8]), &device);
        let y = Tensor::<2, Int>::from_data(TensorData::new(vec![1i64; 16], [2, 8]), &device);
        let batches = vec![(x, y)];

        let before = evaluate(&model, &batches);
        let train_cfg = TrainConfig {
            epochs: 6,
            eval_interval: 1,
            ..Default::default()
        };
        let (model, history) = train(model, &batches, &batches, &train_cfg);
        let after = evaluate(&model, &batches);

        assert!(
            after < before,
            "训练后损失应下降: before={before}, after={after}"
        );
        assert!(history.train_loss.len() >= 6);
    }
}
