//! 4.3 带 GELU 激活的前馈网络。
//!
//! 前馈网络（FFN）是 Transformer 块里"干活"的部分：它把每个位置的向量先升维到 4 倍，
//! 过一次非线性，再降回原维度。注意它**逐位置独立**作用，不与其它 token 交互——
//! 跨 token 的信息交换全部由注意力负责。
//!
//! ## 为什么是 GELU 而不是 ReLU
//!
//! ReLU 在 0 处硬拐弯，负值全部砍掉，梯度信息在负区间彻底消失。
//! GELU 用高斯累积分布函数做平滑门控：
//!
//! ```text
//! GELU(x) = x · Φ(x)          Φ 是标准正态的 CDF
//!         ≈ 0.5·x·(1 + tanh(√(2/π)(x + 0.044715x³)))   ← 实际使用的近似式
//! ```
//!
//! 效果：GELU 在负值区不是 0 而是一条平滑的小尾巴，允许少量负激活通过，
//! 梯度更连续。GPT-2 用的是上面的 tanh 近似式（为了算得快）。

use burn::module::Module;
use burn::nn::{Gelu, Linear, LinearConfig};
use burn::tensor::Tensor;

/// 逐位置前馈网络：`Linear(d → 4d) → GELU → Linear(4d → d)`
#[derive(Module, Debug)]
pub struct FeedForward {
    pub(crate) fc1: Linear,
    pub(crate) fc2: Linear,
    gelu: Gelu,
    d_ff: usize,
}

impl FeedForward {
    /// GPT-2 的隐藏维度固定为 `4 × emb_dim`。
    pub fn new(emb_dim: usize, device: &burn::tensor::Device) -> Self {
        let d_ff = 4 * emb_dim;
        Self {
            fc1: LinearConfig::new(emb_dim, d_ff).init(device),
            fc2: LinearConfig::new(d_ff, emb_dim).init(device),
            // GPT-2 用 tanh 近似版；Burn 的 Gelu 模块直接支持
            gelu: Gelu::new_approximate(),
            d_ff,
        }
    }

    /// `[batch, seq_len, emb_dim]` → 同形状
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let x = self.fc1.forward(x); // 升维到 4 倍
        let x = self.gelu.forward(x); // 非线性
        self.fc2.forward(x) // 降回原维度
    }

    pub fn d_ff(&self) -> usize {
        self.d_ff
    }
}

/// 手动实现 GELU 的 tanh 近似式，用于和 `Gelu::new_approximate()` 对照验证。
pub fn gelu_tanh_manual(x: Tensor<3>) -> Tensor<3> {
    const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
    let inner = (x.clone() + x.clone().powf_scalar(3.0) * 0.044715) * SQRT_2_OVER_PI;
    x * (inner.tanh() + 1.0) * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, TensorData};

    #[test]
    fn ff_preserves_shape() {
        let device = Device::default();
        let ff = FeedForward::new(32, &device);
        let x = Tensor::<3>::zeros([2, 5, 32], &device);
        assert_eq!(ff.forward(x).dims(), [2, 5, 32]);
        assert_eq!(ff.d_ff(), 128);
    }

    #[test]
    fn manual_gelu_matches_burn() {
        let device = Device::default();
        let x = Tensor::<3>::from_data(
            TensorData::new(
                vec![3.0f32, 1.0, 0.5, 0.0, -0.5, -1.0, -3.0],
                [1, 1, 7],
            ),
            &device,
        );
        let a = gelu_tanh_manual(x.clone()).to_data().try_to_vec::<f32>().unwrap();
        let b = Gelu::new_approximate()
            .forward(x)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }

    #[test]
    fn gelu_keeps_negative_values_alive() {
        let device = Device::default();
        let x = Tensor::<3>::from_data(TensorData::new(vec![-1.0f32, -0.5, 0.5], [1, 1, 3]), &device);
        let out = gelu_tanh_manual(x).to_data().try_to_vec::<f32>().unwrap();
        // ReLU 会给出 0, 0, 0.5；GELU 在负区间保留了小幅负值
        assert!(out[0] < 0.0 && out[0] > -0.2, "-1.0 处的 GELU ≈ -0.158，实际 {}", out[0]);
        assert!(out[1] < 0.0);
        assert!(out[2] > 0.0);
    }

    #[test]
    fn ff_is_position_wise() {
        let device = Device::default();
        let ff = FeedForward::new(8, &device);
        // 同一个向量重复两次，输出也应相同（前馈网络不跨位置）
        let row = Tensor::<3>::from_data(TensorData::new(vec![0.1f32; 8], [1, 1, 8]), &device);
        let twice = Tensor::cat(vec![row.clone(), row], 1);
        let out = ff.forward(twice).to_data().try_to_vec::<f32>().unwrap();
        let (a, b) = out.split_at(8);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }
}
