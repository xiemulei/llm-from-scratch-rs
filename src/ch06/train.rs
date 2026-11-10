//! 6.7 – 6.8：分类微调循环与评估。


use burn::optim::{AdamWConfig, GradientsParams};


use crate::ch05::loss::cross_entropy_classes;
use crate::ch06::classifier::SpamClassifier;
use crate::ch06::data::Batch;

pub struct ClassifyConfig {
    pub epochs: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub eval_interval: usize,
}

impl Default for ClassifyConfig {
    fn default() -> Self {
        // 书中 6.7：lr 5e-5、weight decay 0.1、5 个 epoch
        Self { epochs: 5, lr: 5e-5, weight_decay: 0.1, eval_interval: 10 }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Metrics {
    pub loss: f32,
    pub accuracy: f32,
}

#[derive(Debug, Default)]
pub struct History {
    pub train: Vec<(usize, Metrics)>,
    pub val: Vec<(usize, Metrics)>,
}

/// 计算一批数据的平均损失和准确率。
pub fn evaluate(model: &SpamClassifier, batches: &[Batch]) -> Metrics {
    if batches.is_empty() {
        return Metrics::default();
    }
    let mut loss_sum = 0.0f32;
    let mut correct = 0usize;
    let mut total = 0usize;

    for b in batches {
        let logits = model.forward(b.inputs.clone());
        loss_sum += cross_entropy_classes(logits.clone(), b.targets.clone()).into_scalar::<f32>();

        // 取 argmax 和标签在主机端比较（batch 很小，读回代价可以忽略）
        let pred = logits.argmax(1).to_data().try_to_vec::<i32>().unwrap();
        let target = b.targets.to_data().try_to_vec::<i32>().unwrap();
        for (p, t) in pred.iter().zip(target.iter()) {
            if p == t {
                correct += 1;
            }
            total += 1;
        }
    }

    Metrics {
        loss: loss_sum / batches.len() as f32,
        accuracy: correct as f32 / total.max(1) as f32,
    }
}

/// 微调主循环。
pub fn fine_tune(
    mut model: SpamClassifier,
    train: &[Batch],
    val: &[Batch],
    cfg: &ClassifyConfig,
) -> (SpamClassifier, History) {
    let mut optim = AdamWConfig::new()
        .with_weight_decay(cfg.weight_decay as f32)
        .init();

    let mut history = History::default();
    let mut global_step = 0usize;
    let total_steps = cfg.epochs * train.len();

    println!(
        "开始微调：{} epoch × {} 步 = {} 步，可训练参数 {}",
        cfg.epochs,
        train.len(),
        total_steps,
        model.trainable_params()
    );

    for epoch in 1..=cfg.epochs {
        for b in train {
            let logits = model.forward(b.inputs.clone());
            let loss = cross_entropy_classes(logits, b.targets.clone());
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(cfg.lr, model, grads);

            global_step += 1;
            if global_step % cfg.eval_interval == 0 || global_step == total_steps {
                let train_m = evaluate(&model, &train[..1]);
                let val_m = evaluate(&model, val);
                println!(
                    "  epoch {epoch}/{:<2} step {global_step:>4}/{total_steps:<4} 单批损失 {:.4} | val loss {:.4} acc {:.3}",
                    cfg.epochs, train_m.loss, val_m.loss, val_m.accuracy
                );
                history.train.push((global_step, train_m));
                history.val.push((global_step, val_m));
            }
        }
    }

    (model, history)
}

/// 对单条文本做预测，返回 (标签, 置信度)。0 = 正常，1 = 垃圾。
pub fn classify(model: &SpamClassifier, batch: Batch) -> Vec<(usize, f32)> {
    let logits = model.forward(batch.inputs);

    // 手动 softmax（只有两个类，简单写）
    let vals = logits.to_data().try_to_vec::<f32>().unwrap();
    let n = batch.targets.dims()[0];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let (a, b) = (vals[i * 2], vals[i * 2 + 1]);
        let m = a.max(b);
        let (ea, eb) = ((a - m).exp(), (b - m).exp());
        let p1 = eb / (ea + eb);
        out.push(if p1 > 0.5 { (1, p1) } else { (0, 1.0 - p1) });
    }
    out
}
