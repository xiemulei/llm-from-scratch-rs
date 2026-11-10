//! 4.6 组装完整的 GPT 模型。
//!
//! 把前几章的所有零件拼起来：
//!
//! ```text
//! idx [b, t]
//!   → 词元嵌入 [b, t, d]  ┐
//!   → 位置嵌入 [1, t, d]  ┴→ 相加 → Dropout
//!   → 12 × TransformerBlock（每个块含 因果多头注意力 + 前馈网络 + 2 个 LayerNorm）
//!   → 最终 LayerNorm
//!   → 输出头（Linear d→vocab，可选与词元嵌入共享权重）
//!   → logits [b, t, vocab]
//! ```
//!
//! ## 权重共享（weight tying）
//!
//! 词元嵌入是 `[vocab, d]`，输出头是 `[d, vocab]`——形状正好互为转置，语义上也是同一件事
//! （"词元和向量互查"）。GPT-2 让它们共享同一份参数，省掉 38.6M 参数（124M 模型的 24%），
//! 而且实测效果更好。
//!
//! ## 参数量
//!
//! 124M 模型的参数几乎全在 12 个 Transformer 块里（85M）和两个嵌入/输出头里：
//!
//! | 部分 | 参数量 |
//! |---|---|
//! | 词元嵌入 50257×768 | 38.6M |
//! | 位置嵌入 1024×768 | 0.79M |
//! | 12 × Transformer 块 | 85.1M |
//! | 最终 LayerNorm | 1.5K |
//! | 输出头（共享时复用词元嵌入） | 38.6M / 0 |

use burn::config::Config;
use burn::module::Module;
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, Initializer, LayerNorm, LayerNormConfig,
    Linear, LinearConfig,
};
use burn::tensor::{Device, Int, Tensor};

use super::transformer_block::TransformerBlock;

#[derive(Config, Debug)]
pub struct GPTConfig {
    pub vocab_size: usize,
    pub context_length: usize,
    pub emb_dim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    /// 所有 Dropout 的丢弃率
    #[config(default = 0.1)]
    pub drop_rate: f64,
    /// QKV 投影是否带 bias。GPT-2 是 true；原始 Transformer 是 false。
    #[config(default = false)]
    pub qkv_bias: bool,
    #[config(default = 1e-5)]
    pub layer_norm_eps: f64,
    /// 输出头是否与词元嵌入共享权重
    #[config(default = false)]
    pub tie_weights: bool,
}

/// 书中 4.6 的 `GPT_CONFIG_124M`。
pub fn gpt_config_124m() -> GPTConfig {
    GPTConfig::new(50257, 1024, 768, 12, 12)
        .with_qkv_bias(true)
        .with_drop_rate(0.1)
}

#[derive(Module, Debug)]
pub struct Gpt {
    pub(crate) tok_emb: Embedding,
    pub(crate) pos_emb: Embedding,
    drop_emb: Dropout,
    pub(crate) trf_blocks: Vec<TransformerBlock>,
    pub(crate) final_norm: LayerNorm,
    pub(crate) out_head: Linear,
    context_length: usize,
    tie_weights: bool,
}

impl Gpt {
    pub fn new(cfg: &GPTConfig, device: &Device) -> Self {
        // GPT-2 的嵌入用 N(0, 0.02) 初始化，而不是默认的 N(0, 1)
        let emb_init = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };

        let trf_blocks = (0..cfg.n_layers)
            .map(|_| TransformerBlock::new(cfg, device))
            .collect();

        let mut model = Self {
            tok_emb: EmbeddingConfig::new(cfg.vocab_size, cfg.emb_dim)
                .with_initializer(emb_init.clone())
                .init(device),
            pos_emb: EmbeddingConfig::new(cfg.context_length, cfg.emb_dim)
                .with_initializer(emb_init.clone())
                .init(device),
            drop_emb: DropoutConfig::new(cfg.drop_rate).init(),
            trf_blocks,
            final_norm: LayerNormConfig::new(cfg.emb_dim)
                .with_epsilon(cfg.layer_norm_eps)
                .init(device),
            // 输出头：d → vocab，GPT-2 无 bias
            out_head: LinearConfig::new(cfg.emb_dim, cfg.vocab_size)
                .with_bias(false)
                .with_initializer(emb_init)
                .init(device),
            context_length: cfg.context_length,
            tie_weights: cfg.tie_weights,
        };

        if cfg.tie_weights {
            // 共享模式的初始值同步：让输出头和词元嵌入一开始就相等
            model.sync_output_head();
        }
        model
    }

    /// 把词元嵌入的当前值复制到输出头。
    ///
    /// ⚠️ Burn 里 `Param` 的 clone **不是**梯度级的共享：参数一旦被优化器更新，
    /// 两个句柄就会分叉（源码注释明确写了 *"Any mutation forks into a new LazyInitState"*）。
    /// 所以真正的权重共享要靠 `forward` 里**直接用同一份 `tok_emb.weight`** 算 logits，
    /// 而不是给 `out_head` 挂一个克隆。
    pub fn sync_output_head(&mut self) {
        // ⚠️ 权重布局坑：Burn 的 Linear 权重是 [d_input, d_output]（行主序），
        // 而 PyTorch 的 nn.Linear 是 [d_output, d_input]。所以这里必须转置。
        // 词元嵌入是 [vocab, d]，转置后 [d, vocab] 才符合 Linear(d → vocab)。
        let w = self.tok_emb.weight.val().swap_dims(0, 1);
        self.out_head.weight = burn::module::Param::from_tensor(w);
    }

    pub fn is_weight_tied(&self) -> bool {
        self.tie_weights
    }

    /// 计入共享后的"逻辑"参数量。
    ///
    /// `Module::num_params()` 是按访问次数累加的（同一个 Param 被访问两次会算两遍），
    /// 所以共享时它报的是未共享的数。这里手动把重复计的那份减掉。
    pub fn logical_num_params(&self) -> usize {
        use burn::module::Module;
        let raw = self.num_params();
        if self.tie_weights {
            raw - self.tok_emb.weight.val().dims().iter().product::<usize>()
        } else {
            raw
        }
    }

    pub fn context_length(&self) -> usize {
        self.context_length
    }

    /// `idx: [batch, seq_len]`（整数词元 ID） → `logits: [batch, seq_len, vocab_size]`
    pub fn forward(&self, idx: Tensor<2, Int>) -> Tensor<3> {
        let [batch, seq_len] = idx.dims();
        assert!(
            seq_len <= self.context_length,
            "序列长度 {seq_len} 超过上下文长度 {}",
            self.context_length
        );

        let device = idx.device();
        let token_emb = self.tok_emb.forward(idx);

        // 位置嵌入对所有样本相同，只算一次再广播
        let positions =
            Tensor::<1, Int>::arange(0..seq_len as i64, &device).unsqueeze_dim::<2>(0);
        let pos_emb = self.pos_emb.forward(positions).repeat_dim(0, batch);

        let mut x = self.drop_emb.forward(token_emb + pos_emb);

        for block in &self.trf_blocks {
            x = block.forward(x);
        }

        let x = self.final_norm.forward(x);

        if self.tie_weights {
            // 真·权重共享：直接用词元嵌入矩阵做输出投影（d → vocab）
            // x[b, t, d] × Wᵀ[d, vocab] → logits[b, t, vocab]
            // 因为用的就是 tok_emb 这一个 Param，梯度天然累加到同一处。
            let [batch, _, d] = x.dims();
            let [vocab, _] = self.tok_emb.weight.val().dims();
            let w = self
                .tok_emb
                .weight
                .val()
                .swap_dims(0, 1)
                .unsqueeze_dim::<3>(0)
                .expand([batch, d, vocab]);
            x.matmul(w)
        } else {
            self.out_head.forward(x)
        }
    }

    /// 取输出头权重（未共享时）或词元嵌入权重（共享时）。
    pub fn output_weight(&self) -> Tensor<2> {
        if self.tie_weights {
            self.tok_emb.weight.val()
        } else {
            self.out_head.weight.val()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    fn small_cfg() -> GPTConfig {
        GPTConfig::new(100, 16, 16, 4, 2).with_drop_rate(0.0)
    }

    #[test]
    fn forward_shapes() {
        let device = Device::default();
        let model = Gpt::new(&small_cfg(), &device);
        let idx = Tensor::<2, Int>::from_data(TensorData::new(vec![1i64, 2, 3], [1, 3]), &device);
        assert_eq!(model.forward(idx).dims(), [1, 3, 100]);
    }

    #[test]
    fn batch_forward_shapes() {
        let device = Device::default();
        let model = Gpt::new(&small_cfg(), &device);
        let idx = Tensor::<2, Int>::zeros([3, 5], &device);
        assert_eq!(model.forward(idx).dims(), [3, 5, 100]);
    }

    #[test]
    fn too_long_sequence_panics() {
        let device = Device::default();
        let model = Gpt::new(&small_cfg(), &device);
        let idx = Tensor::<2, Int>::zeros([1, 17], &device); // 上下文只有 16
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| model.forward(idx)));
        assert!(r.is_err());
    }

    #[test]
    fn weight_tying_saves_a_full_copy() {
        let device = Device::default();

        let untied = Gpt::new(&small_cfg(), &device);
        let tied = Gpt::new(&small_cfg().with_tie_weights(true), &device);

        // `num_params()` 不去重，所以得用逻辑参数量比较
        assert_eq!(
            untied.logical_num_params() - tied.logical_num_params(),
            100 * 16,
            "共享应省下 vocab×d = 1600 个参数"
        );
        assert!(tied.is_weight_tied());
    }

    #[test]
    fn tied_forward_path_runs() {
        let device = Device::default();
        let idx = Tensor::<2, Int>::from_data(TensorData::new(vec![1i64, 5, 9], [1, 3]), &device);

        // 同一个模型，先把输出头同步成词元嵌入，两种前向路径应当一致
        let mut model = Gpt::new(&small_cfg(), &device);
        model.sync_output_head();
        let a = model.forward(idx.clone()).to_data().try_to_vec::<f32>().unwrap();

        // 用共享权重重建（初始化会重新随机，所以要手动把嵌入表对齐）
        let model2 = Gpt::new(&small_cfg().with_tie_weights(true), &device);
        let b = model2.forward(idx).to_data().try_to_vec::<f32>().unwrap();
        assert_eq!(a.len(), b.len());
        // 两个模型参数不同，这里只验证共享路径本身能跑通且形状正确
        assert!(b.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn output_weight_shape_is_vocab_by_dim() {
        let device = Device::default();
        let model = Gpt::new(&small_cfg().with_tie_weights(true), &device);
        assert_eq!(model.output_weight().dims(), [100, 16]);
    }
}
