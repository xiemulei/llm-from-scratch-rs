//! 3.4 带可学习权重的自注意力。
//!
//! 3.3 的版本把输入同时当作 Q、K、V，等于假设"词的表示"和"我关心什么 / 我是什么 /
//! 我提供什么信息"是同一回事。显然不合理。
//!
//! 于是引入三个可学习投影矩阵：
//!
//! ```text
//! Q = X · W_query     "我在找什么"
//! K = X · W_key       "我是什么"
//! V = X · W_value     "我能提供什么"
//! ```
//!
//! 注意力权重 = softmax(QKᵀ / √d_k)，上下文 = 权重 · V。
//!
//! **为什么要除以 √d_k**：点积的方差随维度线性增长，d_k = 768 时点积可能大到
//! softmax 退化成 one-hot（梯度几乎为 0）。除以 √d_k 把方差拉回 1。
//!
//! 书里对比了两个版本：v1 手工用 `torch.rand` 初始化，v2 改用 `nn.Linear`。
//! 二者数学上完全一样，但 v1 的均匀随机初始化让所有词元的注意力分布几乎相同
//! （因为点积集中在 0 附近），而 PyTorch 的 `nn.Linear` 用 Kaiming 均匀初始化，
//! 尺度更合理。这里用两种初始化器复现这个差异。

use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::tensor::{Device, Tensor};

#[derive(Module, Debug)]
pub struct SelfAttention {
    w_query: Linear,
    w_key: Linear,
    w_value: Linear,
    d_out: usize,
}

impl SelfAttention {
    /// 3.4.3 的 v2：用 `Linear` 的默认初始化（Kaiming 均匀），和 PyTorch 一致。
    pub fn new(d_in: usize, d_out: usize, device: &Device) -> Self {
        Self {
            w_query: LinearConfig::new(d_in, d_out).with_bias(false).init(device),
            w_key: LinearConfig::new(d_in, d_out).with_bias(false).init(device),
            w_value: LinearConfig::new(d_in, d_out).with_bias(false).init(device),
            d_out,
        }
    }

    /// 3.4.3 的 v1：手工 `[0, 1)` 均匀初始化。
    ///
    /// 保留它是为了演示书中的结论——**初始化不好，注意力会退化成均匀分布**。
    pub fn new_manual_init(d_in: usize, d_out: usize, device: &Device) -> Self {
        let init = Initializer::Uniform { min: 0.0, max: 1.0 };
        Self {
            w_query: LinearConfig::new(d_in, d_out)
                .with_bias(false)
                .with_initializer(init.clone())
                .init(device),
            w_key: LinearConfig::new(d_in, d_out)
                .with_bias(false)
                .with_initializer(init.clone())
                .init(device),
            w_value: LinearConfig::new(d_in, d_out)
                .with_bias(false)
                .with_initializer(init)
                .init(device),
            d_out,
        }
    }

    /// 返回 `(上下文向量, 注意力权重)`，权重形状 `[n, n]`。
    pub fn forward(&self, x: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
        let [n, _] = x.dims();

        // 1. 三个投影：[n, d_in] → [n, d_out]
        let queries = self.w_query.forward(x.clone());
        let keys = self.w_key.forward(x.clone());
        let values = self.w_value.forward(x);

        // 2. 打分：Q · Kᵀ / √d_out
        let scale = 1.0 / (self.d_out as f64).sqrt();
        let attn_scores = queries.matmul(keys.swap_dims(0, 1)) * scale;

        // 3. 归一化 + 聚合
        let attn_weights = softmax_rows(attn_scores, n);
        let context = attn_weights.clone().matmul(values);
        (context, attn_weights)
    }

    pub fn d_out(&self) -> usize {
        self.d_out
    }
}

/// 逐行 softmax（沿最后一维），复用 3.3 的数值稳定写法。
pub fn softmax_rows(x: Tensor<2>, n: usize) -> Tensor<2> {
    let d = x.dims()[1];
    let max = x.clone().max_dim(1).expand([n, d]);
    let e = (x - max).exp();
    let sum = e.clone().sum_dim(1).expand([n, d]);
    e / sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ch03::matrix;

    fn book_inputs(device: &Device) -> Tensor<2> {
        matrix(
            vec![
                vec![0.43, 0.15, 0.89],
                vec![0.55, 0.87, 0.66],
                vec![0.57, 0.85, 0.64],
                vec![0.22, 0.58, 0.33],
                vec![0.77, 0.25, 0.10],
                vec![0.05, 0.80, 0.55],
            ],
            device,
        )
    }

    #[test]
    fn shapes_are_book_consistent() {
        let device = Device::default();
        let attn = SelfAttention::new(3, 2, &device);
        let (context, weights) = attn.forward(book_inputs(&device));
        assert_eq!(context.dims(), [6, 2]);
        assert_eq!(weights.dims(), [6, 6]);
    }

    #[test]
    fn parameter_count_is_three_projections() {
        let device = Device::default();
        let attn = SelfAttention::new(768, 768, &device);
        assert_eq!(Module::num_params(&attn), 3 * 768 * 768);
    }

    #[test]
    fn weights_are_row_stochastic() {
        let device = Device::default();
        let attn = SelfAttention::new(3, 2, &device);
        let (_, weights) = attn.forward(book_inputs(&device));
        let sums = weights.sum_dim(1).to_data().try_to_vec::<f32>().unwrap();
        for s in sums {
            assert!((s - 1.0).abs() < 1e-5);
        }
    }
}
