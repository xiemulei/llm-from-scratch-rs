//! 3.5 因果（掩码）自注意力。
//!
//! 语言模型是自回归的：预测第 t 个词时，只能看 0..t，绝不能看到 t+1。
//! 但 3.4 的注意力让每个位置都能看到全部输入——训练时会"偷看答案"。
//!
//! 解决办法：在 softmax **之前**把上三角（未来位置）的分数设成 `-∞`。
//! softmax 之后这些位置的权重自然是 0，未来的信息就进不来了。
//!
//! ```text
//! scores            mask              masked scores        softmax
//! ● ● ●            ● ○ ○             ● -∞ -∞             1.00  0    0
//! ● ● ●     →      ● ● ○      →      ●  ●  -∞      →      .55 .45    0
//! ● ● ●            ● ● ●             ●  ●   ●             .33 .33 .33
//! ```
//!
//! 为什么用 `-∞` 而不是 0：softmax 里 `exp(0) = 1` 仍然贡献权重，必须用 `-∞`
//! 才能让它彻底消失（`exp(-∞) = 0`）。
//!
//! 注意 Burn 的 `Dropout` 只在 `device.is_autodiff()` 时才真正丢弃元素
//! （和 PyTorch 的 `model.train()` 语义一致），纯推理设备上它是恒等映射。

use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::{Bool, Device, Tensor};

#[derive(Module, Debug)]
pub struct CausalSelfAttention {
    w_query: Linear,
    w_key: Linear,
    w_value: Linear,
    dropout: Dropout,
    d_out: usize,
    context_length: usize,
    dropout_p: f64,
}

impl CausalSelfAttention {
    pub fn new(
        d_in: usize,
        d_out: usize,
        context_length: usize,
        dropout_p: f64,
        device: &Device,
    ) -> Self {
        assert!(
            d_out > 0,
            "输出维度必须大于 0（因果掩码要求每个位置至少能看见自己）"
        );
        Self {
            w_query: LinearConfig::new(d_in, d_out).with_bias(false).init(device),
            w_key: LinearConfig::new(d_in, d_out).with_bias(false).init(device),
            w_value: LinearConfig::new(d_in, d_out).with_bias(false).init(device),
            dropout: DropoutConfig::new(dropout_p).init(),
            d_out,
            context_length,
            dropout_p,
        }
    }

    /// `x: [batch, seq_len, d_in]` → `(context [batch, seq_len, d_out], weights [batch, seq_len, seq_len])`
    pub fn forward(&self, x: Tensor<3>) -> (Tensor<3>, Tensor<3>) {
        let [batch, seq_len, _] = x.dims();
        assert!(
            seq_len <= self.context_length,
            "序列长度 {seq_len} 超过上下文长度 {}",
            self.context_length
        );

        let queries = self.w_query.forward(x.clone());
        let keys = self.w_key.forward(x.clone());
        let device = x.device();
        let values = self.w_value.forward(x);

        let scale = 1.0 / (self.d_out as f64).sqrt();
        let attn_scores = queries.matmul(keys.swap_dims(1, 2)) * scale;

        // 关键的一步：掩掉未来
        let mask = causal_mask(batch, seq_len, &device);
        let attn_scores = attn_scores.mask_fill(mask, f32::NEG_INFINITY);

        let attn_weights = burn::tensor::activation::softmax(attn_scores, 2);
        // 书中的顺序是 softmax → dropout → 加权求和
        let attn_weights = self.dropout.forward(attn_weights);

        let context = attn_weights.clone().matmul(values);
        (context, attn_weights)
    }

    pub fn dropout_p(&self) -> f64 {
        self.dropout_p
    }
}

/// 生成 `[batch, seq_len, seq_len]` 的布尔掩码：`true` 表示"要屏蔽"。
///
/// ⚠️ 注意 Burn 和 PyTorch 的语义差异：Burn 的 `tril_mask` 返回的是**下三角为 false、
/// 上三角为 true** 的掩码（名字叫 "mask"，语义是"待填充的位置"），
/// 正好可以直接丢给 `mask_fill`；而 PyTorch 的 `torch.tril` 返回的是下三角为 true。
/// 所以这里**不需要** `bool_not()`。
pub fn causal_mask(batch: usize, seq_len: usize, device: &Device) -> Tensor<3, Bool> {
    Tensor::<2, Bool>::tril_mask([seq_len, seq_len], 0, device)
        .unsqueeze_dim::<3>(0)
        .expand([batch, seq_len, seq_len])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn future_is_masked_out() {
        let device = Device::default();
        let attn = CausalSelfAttention::new(8, 8, 16, 0.0, &device);
        let x = Tensor::<3>::random(
            [2, 5, 8],
            burn::tensor::Distribution::Default,
            &device,
        );
        let (context, weights) = attn.forward(x);
        assert_eq!(context.dims(), [2, 5, 8]);
        assert_eq!(weights.dims(), [2, 5, 5]);

        // 上三角必须严格为 0
        let w = weights.to_data().try_to_vec::<f32>().unwrap();
        for b in 0..2 {
            for i in 0..5 {
                for j in 0..5 {
                    let v = w[(b * 5 + i) * 5 + j];
                    if j > i {
                        assert!(v.abs() < 1e-6, "位置 {i} 看到了未来的 {j}: {v}");
                    }
                }
            }
        }
    }

    #[test]
    fn rows_still_sum_to_one() {
        let device = Device::default();
        let attn = CausalSelfAttention::new(8, 8, 16, 0.0, &device);
        let x = Tensor::<3>::random([1, 4, 8], burn::tensor::Distribution::Default, &device);
        let (_, weights) = attn.forward(x);
        let sums = weights.sum_dim(2).to_data().try_to_vec::<f32>().unwrap();
        for s in sums {
            assert!((s - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn first_token_attends_only_to_itself() {
        let device = Device::default();
        let attn = CausalSelfAttention::new(4, 4, 8, 0.0, &device);
        let x = Tensor::<3>::zeros([1, 3, 4], &device);
        let (_, weights) = attn.forward(x);
        let w = weights.to_data().try_to_vec::<f32>().unwrap();
        assert!((w[0] - 1.0).abs() < 1e-6);
        assert!(w[1].abs() < 1e-6 && w[2].abs() < 1e-6);
    }
}
