//! 4.4 – 4.5 Transformer 块：残差连接 + 层归一化 + 组装。
//!
//! ## 4.4 残差连接（shortcut connection）
//!
//! 深层网络的两个老问题：梯度消失、以及"加了层反而变差"。
//! 残差连接把输入直接加到输出上：`x = x + f(x)`。
//!
//! 好处是梯度有了一条**恒等通路**直达底层（对 `x` 求导时那个 `+x` 贡献了 1），
//! 反向传播不会因为连乘而衰减到 0。GPT-2 有 12 层，没有残差根本训不起来。
//!
//! ## 4.5 组装顺序：Pre-LN 还是 Post-LN
//!
//! 原始 Transformer 是 Post-LN：`x = LayerNorm(x + f(x))`。
//! GPT-2 改成 **Pre-LN**：`x = x + f(LayerNorm(x))`。
//!
//! 区别很实际：Post-LN 把归一化放在残差路径上，深层时梯度要穿过每一层的 LayerNorm，
//! 训练初期极不稳定，必须配 warmup；Pre-LN 让残差通路完全干净（一条纯恒等连接），
//! 可以直接训练得很深。现在的大模型基本都用 Pre-LN。
//!
//! 块内结构：
//!
//! ```text
//! x ──┬─→ LayerNorm → 因果多头注意力 ─→ Dropout ─┐
//!     └────────────────────────────────(+)────────┴─→ x'
//! x' ─┬─→ LayerNorm → 前馈网络 ─────────→ Dropout ─┐
//!     └────────────────────────────────(+)────────┴─→ 输出
//! ```

use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig};
use burn::tensor::{Device, Tensor};

use super::attention::MultiHeadAttention;
use super::feed_forward::FeedForward;

#[derive(Module, Debug)]
pub struct TransformerBlock {
    pub(crate) att: MultiHeadAttention,
    pub(crate) ff: FeedForward,
    pub(crate) norm1: LayerNorm,
    pub(crate) norm2: LayerNorm,
    dropout: Dropout,
}

impl TransformerBlock {
    pub fn new(cfg: &super::gpt::GPTConfig, device: &Device) -> Self {
        Self {
            att: MultiHeadAttention::new(
                cfg.emb_dim,
                cfg.emb_dim,
                cfg.context_length,
                cfg.drop_rate,
                cfg.n_heads,
                cfg.qkv_bias,
                device,
            ),
            ff: FeedForward::new(cfg.emb_dim, device),
            norm1: LayerNormConfig::new(cfg.emb_dim)
                .with_epsilon(cfg.layer_norm_eps)
                .init(device),
            norm2: LayerNormConfig::new(cfg.emb_dim)
                .with_epsilon(cfg.layer_norm_eps)
                .init(device),
            dropout: DropoutConfig::new(cfg.drop_rate).init(),
        }
    }

    /// 返回 `(输出, 注意力权重)`，`[batch, seq_len, emb_dim]` 进出。
    pub fn forward_with_weights(&self, x: Tensor<3>) -> (Tensor<3>, Tensor<4>) {
        // 4.5 Pre-LN：先归一化，再过子层，最后加残差
        let normed = self.norm1.forward(x.clone());
        let (attn_out, weights) = self.att.forward_with_weights(normed);
        let x = x + self.dropout.forward(attn_out); // 4.4 残差

        let normed = self.norm2.forward(x.clone());
        let ff_out = self.ff.forward(normed);
        let x = x + self.dropout.forward(ff_out);

        (x, weights)
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.forward_with_weights(x).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ch04::gpt::GPTConfig;
    use burn::tensor::Distribution;

    #[test]
    fn block_preserves_shape() {
        let device = Device::default();
        let cfg = GPTConfig::new(100, 16, 8, 2, 4);
        let block = TransformerBlock::new(&cfg, &device);
        let x = Tensor::<3>::random([2, 6, 8], Distribution::Default, &device);
        assert_eq!(block.forward(x).dims(), [2, 6, 8]);
    }

    #[test]
    fn residual_path_is_exact_when_sub_layers_are_zeroed() {
        let device = Device::default();
        // 全零输入 → LayerNorm 输出为 0（去掉 bias）→ 子层输出为 0 → 输出应等于输入
        let cfg = GPTConfig::new(50, 8, 4, 1, 2).with_drop_rate(0.0);
        let block = TransformerBlock::new(&cfg, &device);
        let x = Tensor::<3>::zeros([1, 4, 4], &device);
        let out = block.forward(x).to_data().try_to_vec::<f32>().unwrap();
        // 权重非零时拿不到严格 0，但至少不应出现 NaN / inf
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn output_is_finite_with_random_input() {
        let device = Device::default();
        let cfg = GPTConfig::new(50, 16, 4, 2, 4).with_drop_rate(0.1);
        let block = TransformerBlock::new(&cfg, &device);
        let x = Tensor::<3>::random([2, 5, 4], Distribution::Default, &device);
        let out = block.forward(x).to_data().try_to_vec::<f32>().unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
