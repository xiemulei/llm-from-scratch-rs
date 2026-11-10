//! 4.7 文本生成。
//!
//! 模型输出的是 `[batch, seq_len, vocab]` 的 logits——每个位置对全词表的打分。
//! 生成时只关心**最后一个位置**，把它变成"下一个词元"的采样分布，然后追加回去，
//! 循环 `max_new_tokens` 次。
//!
//! 三种解码策略：
//!
//! | 策略 | 做法 | 特点 |
//! |---|---|---|
//! | 贪心 Greedy | 每次取概率最大的 | 确定、易重复、容易陷入循环 |
//! | 温度 Temperature | 概率除以 T 再采样 | T<1 更保守，T>1 更发散，T=0 等价贪心 |
//! | Top-k | 只在前 k 个候选里按概率采样 | 砍掉长尾，避免生成胡话 |
//!
//! 温度的作用：`softmax(logits / T)`。T→0 时分布退化成 one-hot（贪心），
//! T=1 是原始分布，T→∞ 趋于均匀分布。

use burn::tensor::activation::softmax;
use burn::tensor::{Device, Int, Tensor};

use super::gpt::Gpt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sampling {
    /// 贪心：每步取概率最大的词元
    Greedy,
    /// 温度采样（`temperature` 必须 > 0）
    Temperature(f64),
    /// Top-k 采样：先截断到前 k 个候选，再按温度采样
    TopK { k: usize, temperature: f64 },
}

impl Sampling {
    pub fn describe(&self) -> String {
        match self {
            Self::Greedy => "贪心解码".to_string(),
            Self::Temperature(t) => format!("温度采样 (T={t})"),
            Self::TopK { k, temperature } => format!("Top-k 采样 (k={k}, T={temperature})"),
        }
    }
}

/// 自回归生成。
///
/// `idx: [batch, seq_len]` 是提示词，返回 `[batch, seq_len + max_new_tokens]`。
pub fn generate(
    model: &Gpt,
    mut idx: Tensor<2, Int>,
    max_new_tokens: usize,
    sampling: Sampling,
    device: &Device,
) -> Tensor<2, Int> {
    let context_length = model.context_length();

    for _ in 0..max_new_tokens {
        // 位置编码的最大长度是 context_length，超了就得从左边裁掉最早的词元
        let [batch, seq_len] = idx.dims();
        if seq_len > context_length {
            let start = seq_len - context_length;
            idx = idx.slice([0..batch, start..seq_len]);
        }

        let [_, t] = idx.dims();
        let logits = model.forward(idx.clone()); // [b, t, vocab]
        let vocab = logits.dims()[2];

        // 只要最后一个位置的分布
        let last = logits
            .slice([0..batch, (t - 1)..t, 0..vocab])
            .squeeze_dim::<2>(1); // [b, vocab]

        let next = sample_next(last, sampling);
        idx = Tensor::cat(vec![idx, next], 1);
    }

    let _ = device;
    idx
}

/// 从 `[batch, vocab]` 的 logits 采样出 `[batch, 1]` 的下一个词元。
pub fn sample_next(logits: Tensor<2>, sampling: Sampling) -> Tensor<2, Int> {
    let [batch, vocab] = logits.dims();

    match sampling {
        // Burn 的 argmax 保留维度：返回 [b, 1]（不像 numpy 会降维）
        Sampling::Greedy => logits.argmax(1),

        Sampling::Temperature(temperature) => {
            assert!(temperature > 0.0, "温度必须大于 0");
            let probs = softmax(logits / temperature, 1);
            probs.categorical(1)
        }

        Sampling::TopK { k, temperature } => {
            assert!(temperature > 0.0, "温度必须大于 0");
            let k = k.clamp(1, vocab);
            let probs = softmax(logits / temperature, 1);

            // 取第 k 大的概率值作为阈值，比它小的全部清零
            // （等价于书中先排序再保留前 k 个，但不需要知道下标）
            let kth = probs
                .clone()
                .sort_descending(1)
                .slice([0..batch, (k - 1)..k]); // [b, 1]
            let drop_mask = probs.clone().lower_equal(kth.expand([batch, vocab]));
            let filtered = probs.mask_fill(drop_mask, 0.0);

            // 重新归一化，否则 categorical 拿到的不是合法分布
            let sum = filtered.clone().sum_dim(1).expand([batch, vocab]);
            let filtered = filtered / sum;
            filtered.categorical(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    fn logits(rows: Vec<Vec<f32>>, device: &Device) -> Tensor<2> {
        let n = rows.len();
        let m = rows[0].len();
        Tensor::<2>::from_data(
            TensorData::new(rows.into_iter().flatten().collect::<Vec<f32>>(), [n, m]),
            device,
        )
    }

    #[test]
    fn greedy_takes_argmax() {
        let device = Device::default();
        let l = logits(vec![vec![1.0, 5.0, 3.0], vec![9.0, 0.0, 1.0]], &device);
        let out = sample_next(l, Sampling::Greedy)
            .to_data()
            .try_to_vec::<i32>()
            .unwrap();
        assert_eq!(out, vec![1, 0]);
    }

    #[test]
    fn near_zero_temperature_becomes_greedy() {
        let device = Device::default();
        let l = logits(vec![vec![1.0, 5.0, 3.0]], &device);
        let mut hits = [0usize; 3];
        for _ in 0..50 {
            let id = sample_next(l.clone(), Sampling::Temperature(0.01))
                .to_data()
                .try_to_vec::<i32>()
                .unwrap()[0];
            hits[id as usize] += 1;
        }
        assert_eq!(hits[1], 50, "T→0 应几乎退化为贪心");
    }

    #[test]
    fn high_temperature_spreads_samples() {
        let device = Device::default();
        let l = logits(vec![vec![1.0, 2.0, 1.5]], &device);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            let id = sample_next(l.clone(), Sampling::Temperature(5.0))
                .to_data()
                .try_to_vec::<i32>()
                .unwrap()[0];
            seen.insert(id);
        }
        assert!(seen.len() > 1, "高温度应该能采到不同词元");
    }

    #[test]
    fn top_k_never_picks_outside_the_top_k() {
        let device = Device::default();
        // 词元 2 的概率极低，top-k=2 时永远不该被选中
        let l = logits(vec![vec![5.0, 4.9, -5.0]], &device);
        for _ in 0..100 {
            let id = sample_next(l.clone(), Sampling::TopK { k: 2, temperature: 1.0 })
                .to_data()
                .try_to_vec::<i32>()
                .unwrap()[0];
            assert_ne!(id, 2);
        }
    }

    #[test]
    fn top_k_larger_than_vocab_is_clamped() {
        let device = Device::default();
        let l = logits(vec![vec![1.0, 2.0]], &device);
        let _ = sample_next(l, Sampling::TopK { k: 999, temperature: 1.0 });
    }
}
