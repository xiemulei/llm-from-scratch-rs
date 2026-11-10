//! 4.6（注意力部分）：GPT-2 忠实版多头因果注意力。
//!
//! 第 3 章的多头注意力是教学版，没有 bias。GPT-2 的实际结构有两点不同：
//!
//! 1. **QKV 投影带 bias**（`qkv_bias = true`）。这是 GPT-2 相对原始 Transformer 的改动之一，
//!    加载官方权重时必须对齐，否则形状对不上。
//! 2. 输出投影 `out_proj` 也带 bias。
//!
//! 这里复用的是第 3 章那套 `causal_mask`，避免重复实现掩码逻辑。

use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::{Device, Tensor};

use crate::ch03::causal::causal_mask;

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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_in: usize,
        d_out: usize,
        context_length: usize,
        dropout_p: f64,
        n_heads: usize,
        qkv_bias: bool,
        device: &Device,
    ) -> Self {
        assert!(
            d_out % n_heads == 0,
            "d_out ({d_out}) 必须能被 n_heads ({n_heads}) 整除"
        );
        // GPT-2 的做法：qkv 和 out_proj 的 bias 由同一个开关控制
        let proj = |d_in: usize, d_out: usize| {
            LinearConfig::new(d_in, d_out)
                .with_bias(qkv_bias)
                .init(device)
        };
        Self {
            w_query: proj(d_in, d_out),
            w_key: proj(d_in, d_out),
            w_value: proj(d_in, d_out),
            out_proj: proj(d_out, d_out),
            dropout: DropoutConfig::new(dropout_p).init(),
            n_heads,
            head_dim: d_out / n_heads,
            d_out,
            context_length,
        }
    }

    /// `[batch, seq_len, d_in]` → `[batch, seq_len, d_out]`
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.forward_with_weights(x).0
    }

    /// 额外返回每个头的注意力权重 `[batch, n_heads, seq_len, seq_len]`，供可视化。
    pub fn forward_with_weights(&self, x: Tensor<3>) -> (Tensor<3>, Tensor<4>) {
        let [batch, seq_len, _] = x.dims();
        assert!(
            seq_len <= self.context_length,
            "序列长度 {seq_len} 超过上下文长度 {}",
            self.context_length
        );

        let device = x.device();
        let queries = self.w_query.forward(x.clone());
        let keys = self.w_key.forward(x.clone());
        let values = self.w_value.forward(x);

        let q = split_heads(queries, batch, seq_len, self.n_heads, self.head_dim);
        let k = split_heads(keys, batch, seq_len, self.n_heads, self.head_dim);
        let v = split_heads(values, batch, seq_len, self.n_heads, self.head_dim);

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)) * scale;

        let mask = causal_mask(batch, seq_len, &device)
            .unsqueeze_dim::<4>(1)
            .expand([batch, self.n_heads, seq_len, seq_len]);
        let scores = scores.mask_fill(mask, f32::NEG_INFINITY);

        let weights = burn::tensor::activation::softmax(scores, 3);
        let weights = self.dropout.forward(weights);

        let context = weights.clone().matmul(v);
        let context = merge_heads(context, batch, seq_len, self.n_heads, self.head_dim);
        (self.out_proj.forward(context), weights)
    }
}

fn split_heads(x: Tensor<3>, b: usize, t: usize, h: usize, hd: usize) -> Tensor<4> {
    x.reshape([b, t, h, hd]).swap_dims(1, 2)
}

fn merge_heads(x: Tensor<4>, b: usize, t: usize, h: usize, hd: usize) -> Tensor<3> {
    x.swap_dims(1, 2).reshape([b, t, h * hd])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    #[test]
    fn shape_and_mask() {
        let device = Device::default();
        let mha = MultiHeadAttention::new(16, 16, 8, 0.0, 4, true, &device);
        let x = Tensor::<3>::random([2, 5, 16], Distribution::Default, &device);
        let (out, weights) = mha.forward_with_weights(x);
        assert_eq!(out.dims(), [2, 5, 16]);
        assert_eq!(weights.dims(), [2, 4, 5, 5]);

        let w = weights.to_data().try_to_vec::<f32>().unwrap();
        for b in 0..2 {
            for h in 0..4 {
                for i in 0..5 {
                    for j in 0..5 {
                        let idx = ((b * 4 + h) * 5 + i) * 5 + j;
                        if j > i {
                            assert!(w[idx].abs() < 1e-6);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn bias_toggle_changes_parameter_count() {
        let device = Device::default();
        use burn::module::Module;
        let with_bias = MultiHeadAttention::new(16, 16, 8, 0.0, 4, true, &device);
        let no_bias = MultiHeadAttention::new(16, 16, 8, 0.0, 4, false, &device);
        // 4 个投影，每个多出 d_out 个 bias（3×16 + 16 = 64）
        assert_eq!(
            Module::num_params(&with_bias) - Module::num_params(&no_bias),
            4 * 16
        );
    }
}
