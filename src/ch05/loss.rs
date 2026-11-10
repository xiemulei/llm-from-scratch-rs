//! 5.2 – 5.4：损失函数。
//!
//! ## 5.2 交叉熵
//!
//! 语言模型的输出是 `[batch, seq, vocab]` 的 logits，训练目标是"下一个词元"。
//! 关键的一步是**错位**：
//!
//! ```text
//! 输入 : [Every,  effort, moves, you]
//! 目标 : [effort, moves,  you,   ...]   ← 整体左移一位
//! ```
//!
//! 然后对每个位置计算 `-log(p_target)`，全 batch 全位置取平均。
//!
//! 直觉：交叉熵 = "模型给正确答案分配了多少概率"。答案概率 1 时损失 0，
//! 概率趋向 0 时损失趋向无穷。
//!
//! 困惑度（perplexity）= `exp(loss)`，可以理解为"模型在多少个词元之间犹豫"。
//! 随机初始化时 vocab=50257，困惑度约等于词表大小；训练好的 GPT-2 在英文上能到 20~30。

use burn::tensor::activation::log_softmax;
use burn::tensor::{Int, Tensor};

/// 交叉熵损失：logits `[batch, seq, vocab]`，target `[batch, seq]`。
///
/// 返回标量损失。内部做了两步展平 + 取对数概率，数学上等价于
/// `F.cross_entropy(logits.view(-1, vocab), targets.view(-1))`。
pub fn cross_entropy(logits: Tensor<3>, targets: Tensor<2, Int>) -> Tensor<1> {
    let [batch, seq, vocab] = logits.dims();
    let [tb, ts] = targets.dims();
    assert_eq!((batch, seq), (tb, ts), "logits 与 targets 的形状必须对齐");

    // 1. 展平成 [batch*seq, vocab]
    let logits = logits.reshape([batch * seq, vocab]);
    // 2. 每行变成对数概率分布
    let log_probs = log_softmax(logits, 1);
    // 3. 取出目标位置的对数概率：[batch*seq, 1]
    let targets = targets.reshape([batch * seq, 1]);
    let picked = log_probs.gather(1, targets); // [batch*seq, 1]
    // 4. 取负均值
    picked.mean().neg()
}

/// 5.2：把 `(logits, targets)` 错位，得到下一步预测的输入/目标对。
///
/// 训练时我们其实直接复用批次里的 `inputs` / `targets`（第 2 章已经错好位了），
/// 这个函数只是把书中 `logits[:, :-1]` / `targets[:, 1:]` 的做法也演示出来。
pub fn shift_for_next_token(
    logits: Tensor<3>,
    targets: Tensor<2, Int>,
) -> (Tensor<3>, Tensor<2, Int>) {
    let [b, t, v] = logits.dims();
    let logits = logits.slice([0..b, 0..t - 1, 0..v]);
    let targets = targets.slice([0..b, 1..t]);
    (logits, targets)
}

/// 困惑度：损失的指数。
pub fn perplexity(loss: f32) -> f32 {
    loss.exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, TensorData};

    /// 构造"完美预测"：logits 在正确类别上给一个极大的值
    fn perfect_logits(targets: &[i64], vocab: usize, device: &Device) -> Tensor<3> {
        let seq = targets.len();
        let mut data = vec![0.0f32; seq * vocab];
        for (i, &t) in targets.iter().enumerate() {
            data[i * vocab + t as usize] = 100.0;
        }
        Tensor::<3>::from_data(TensorData::new(data, [1, seq, vocab]), device)
    }

    #[test]
    fn perfect_prediction_has_near_zero_loss() {
        let device = Device::default();
        let targets = vec![1i64, 3, 0];
        let logits = perfect_logits(&targets, 4, &device);
        let t = Tensor::<2, Int>::from_data(TensorData::new(targets, [1, 3]), &device);
        let loss = cross_entropy(logits, t).into_scalar::<f32>();
        assert!(loss < 1e-3, "完美预测的损失应接近 0，实际 {loss}");
    }

    #[test]
    fn uniform_prediction_loses_log_vocab() {
        let device = Device::default();
        let vocab = 8;
        // 全 0 的 logits → softmax 均匀分布 → 每项概率 1/vocab → 损失 = ln(vocab)
        let logits = Tensor::<3>::zeros([1, 5, vocab], &device);
        let t = Tensor::<2, Int>::zeros([1, 5], &device);
        let loss = cross_entropy(logits, t).into_scalar::<f32>();
        assert!(
            (loss - (vocab as f32).ln()).abs() < 1e-4,
            "均匀分布损失应为 ln(vocab)={}，实际 {loss}",
            (vocab as f32).ln()
        );
    }

    #[test]
    fn perplexity_of_uniform_model_equals_vocab_size() {
        // 随机初始化时困惑度 ≈ 词表大小，这是判断训练是否真的开始的一个信号
        let vocab = 1000;
        let p = perplexity((vocab as f32).ln());
        assert!((p - vocab as f32).abs() < 1e-2);
    }

    #[test]
    fn shift_matches_book_slicing() {
        let device = Device::default();
        let logits = Tensor::<3>::zeros([1, 5, 3], &device);
        let targets = Tensor::<2, Int>::from_data(
            TensorData::new(vec![0i64, 1, 2, 1, 0], [1, 5]),
            &device,
        );
        let (l, t) = shift_for_next_token(logits, targets);
        assert_eq!(l.dims(), [1, 4, 3]);
        assert_eq!(t.dims(), [1, 4]);
        assert_eq!(t.to_data().try_to_vec::<i32>().unwrap(), vec![1, 2, 1, 0]);
    }
}
