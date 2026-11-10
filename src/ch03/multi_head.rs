//! 3.6 多头注意力。
//!
//! 单头注意力只能学**一种**关注模式。但语言里同时存在多种关系：
//! 语法依赖、指代关系、局部搭配、篇章结构……多头让模型并行地学多套关注模式。
//!
//! 两种实现方式，书中都讲了：
//!
//! **3.6.1 朴素堆叠**：训练 n 个独立的自注意力，把输出拼接。
//! 参数是 `n_heads × d_in × d_out`——因为每个头都会投影到完整的 `d_out`。
//!
//! **3.6.2 权重切分（正确做法）**：只用三组投影，投影后把 `d_out` 平均切成
//! `n_heads` 份，每份维度 `head_dim = d_out / n_heads`。参数量降到 `1/n_heads`。
//!
//! ```text
//!                     ┌─ head_0: [b, t, head_dim] ─┐
//! X → W_q/W_k/W_v →   ├─ head_1: [b, t, head_dim] ─┤ → concat → W_out → [b, t, d_out]
//!                     └─ head_n: [b, t, head_dim] ─┘
//! ```

use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::Tensor;

use super::causal::{causal_mask, CausalSelfAttention};

/// 3.6.2：高效的多头注意力（GPT-2 采用的形式）。
#[derive(Module, Debug)]
pub struct MultiHeadAttention {
    pub(crate) w_query: Linear,
    pub(crate) w_key: Linear,
    pub(crate) w_value: Linear,
    pub(crate) out_proj: Linear,
    dropout: Dropout,
    n_heads: usize,
    head_dim: usize,
    d_out: usize,
    context_length: usize,
}

impl MultiHeadAttention {
    pub fn new(
        d_in: usize,
        d_out: usize,
        context_length: usize,
        dropout_p: f64,
        n_heads: usize,
        device: &burn::tensor::Device,
    ) -> Self {
        assert!(
            d_out % n_heads == 0,
            "d_out ({d_out}) 必须能被 n_heads ({n_heads}) 整除，否则无法均分给每个头"
        );
        Self {
            w_query: LinearConfig::new(d_in, d_out).with_bias(false).init(device),
            w_key: LinearConfig::new(d_in, d_out).with_bias(false).init(device),
            w_value: LinearConfig::new(d_in, d_out).with_bias(false).init(device),
            // 拼接后需要一次线性变换把各头的结果融合
            out_proj: LinearConfig::new(d_out, d_out).with_bias(false).init(device),
            dropout: DropoutConfig::new(dropout_p).init(),
            n_heads,
            head_dim: d_out / n_heads,
            d_out,
            context_length,
        }
    }

    pub fn n_heads(&self) -> usize {
        self.n_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// `x: [batch, seq_len, d_in]` → `[batch, seq_len, d_out]`
    ///
    /// 第二个返回值是每个头的注意力权重，形状 `[batch, n_heads, seq_len, seq_len]`，
    /// 留作可视化用。
    pub fn forward(&self, x: Tensor<3>) -> (Tensor<3>, Tensor<4>) {
        let [batch, seq_len, _] = x.dims();
        assert!(
            seq_len <= self.context_length,
            "序列长度 {seq_len} 超过上下文长度 {}",
            self.context_length
        );

        // 1. 投影：[b, t, d_in] → [b, t, d_out]
        let queries = self.w_query.forward(x.clone());
        let keys = self.w_key.forward(x.clone());
        let device = x.device();
        let values = self.w_value.forward(x);

        // 2. 切头：[b, t, d_out] → [b, t, h, hd] → [b, h, t, hd]
        let queries = split_heads(queries, batch, seq_len, self.n_heads, self.head_dim);
        let keys = split_heads(keys, batch, seq_len, self.n_heads, self.head_dim);
        let values = split_heads(values, batch, seq_len, self.n_heads, self.head_dim);

        // 3. 缩放点积注意力，逐头并行
        //    [b, h, t, hd] × [b, h, hd, t] → [b, h, t, t]
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_scores = queries.matmul(keys.swap_dims(2, 3)) * scale;

        let mask = causal_mask(batch, seq_len, &device)
            .unsqueeze_dim::<4>(1)
            .expand([batch, self.n_heads, seq_len, seq_len]);
        let attn_scores = attn_scores.mask_fill(mask, f32::NEG_INFINITY);

        let attn_weights = burn::tensor::activation::softmax(attn_scores, 3);
        let attn_weights = self.dropout.forward(attn_weights);

        // 4. 聚合 + 拼回：[b, h, t, hd] → [b, t, h, hd] → [b, t, d_out]
        let context = attn_weights.clone().matmul(values);
        let context = merge_heads(context, batch, seq_len, self.n_heads, self.head_dim);

        // 5. 输出投影，让各头的信息互相混合
        (self.out_proj.forward(context), attn_weights)
    }
}

/// 3.6.1：朴素堆叠版——每个头都是一个完整独立的因果自注意力。
///
/// 多花了 `n_heads` 倍的参数，但逻辑最直白，适合对照理解。
#[derive(Module, Debug)]
pub struct MultiHeadAttentionWrapper {
    heads: Vec<CausalSelfAttention>,
    pub(crate) out_proj: Linear,
    n_heads: usize,
    d_out: usize,
}

impl MultiHeadAttentionWrapper {
    pub fn new(
        d_in: usize,
        d_out: usize,
        context_length: usize,
        dropout_p: f64,
        n_heads: usize,
        device: &burn::tensor::Device,
    ) -> Self {
        let heads = (0..n_heads)
            .map(|_| CausalSelfAttention::new(d_in, d_out, context_length, dropout_p, device))
            .collect();
        Self {
            heads,
            out_proj: LinearConfig::new(n_heads * d_out, d_out)
                .with_bias(false)
                .init(device),
            n_heads,
            d_out,
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, seq_len, _] = x.dims();
        // 每个头独立计算，然后沿最后一维拼接
        let contexts: Vec<Tensor<3>> = self
            .heads
            .iter()
            .map(|head| head.forward(x.clone()).0)
            .collect();
        let concatenated = Tensor::cat(contexts, 2); // [b, t, n_heads * d_out]
        debug_assert_eq!(concatenated.dims(), [batch, seq_len, self.n_heads * self.d_out]);
        self.out_proj.forward(concatenated)
    }
}

/// `[b, t, d_out]` → `[b, h, t, head_dim]`
fn split_heads(x: Tensor<3>, batch: usize, seq_len: usize, n_heads: usize, head_dim: usize) -> Tensor<4> {
    x.reshape([batch, seq_len, n_heads, head_dim])
        .swap_dims(1, 2)
}

/// `[b, h, t, head_dim]` → `[b, t, d_out]`
fn merge_heads(x: Tensor<4>, batch: usize, seq_len: usize, n_heads: usize, head_dim: usize) -> Tensor<3> {
    x.swap_dims(1, 2)
        .reshape([batch, seq_len, n_heads * head_dim])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::Module;
    use burn::tensor::{Device, Distribution};

    #[test]
    fn output_shape_is_preserved() {
        let device = Device::default();
        let mha = MultiHeadAttention::new(32, 32, 16, 0.0, 4, &device);
        let x = Tensor::<3>::random([2, 5, 32], Distribution::Default, &device);
        let (out, weights) = mha.forward(x);
        assert_eq!(out.dims(), [2, 5, 32]);
        assert_eq!(weights.dims(), [2, 4, 5, 5]);
    }

    #[test]
    fn heads_are_causally_masked() {
        let device = Device::default();
        let mha = MultiHeadAttention::new(16, 16, 8, 0.0, 4, &device);
        let x = Tensor::<3>::random([1, 4, 16], Distribution::Default, &device);
        let (_, weights) = mha.forward(x);
        let w = weights.to_data().try_to_vec::<f32>().unwrap();
        let (h, t) = (4usize, 4usize);
        for hi in 0..h {
            for i in 0..t {
                for j in 0..t {
                    if j > i {
                        assert!(w[(hi * t + i) * t + j].abs() < 1e-6);
                    }
                }
            }
        }
    }

    #[test]
    fn efficient_version_uses_fewer_params() {
        let device = Device::default();
        let d_in = 64;
        let d_out = 64;
        let n_heads = 4;

        let efficient = MultiHeadAttention::new(d_in, d_out, 32, 0.0, n_heads, &device);
        let naive = MultiHeadAttentionWrapper::new(d_in, d_out, 32, 0.0, n_heads, &device);

        let p_efficient = Module::num_params(&efficient);
        let p_naive = Module::num_params(&naive);

        // 高效版 = 3×d² + d²（输出投影）；朴素版 = n×(3×d²) + n·d²（输出投影）
        assert_eq!(p_efficient, 4 * d_in * d_out);
        assert_eq!(p_naive, n_heads * 4 * d_in * d_out);
        assert_eq!(p_naive / p_efficient, n_heads);
    }

    #[test]
    fn divisible_heads_are_enforced() {
        let device = Device::default();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            MultiHeadAttention::new(8, 9, 8, 0.0, 4, &device);
        }));
        assert!(result.is_err(), "9 不能被 4 整除，应当 panic");
    }
}
