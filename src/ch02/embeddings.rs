//! 2.7 – 2.8：词元嵌入与位置编码。
//!
//! 到这里，文本已经变成 `[batch, seq_len]` 的整数矩阵。但 Transformer 只吃浮点数，
//! 所以还需要两步：
//!
//! **2.7 词元嵌入**：查表。词表大小 × 嵌入维度 的一个矩阵，第 i 行就是词元 i 的向量。
//!   这张表是**可学习参数**，预训练过程中不断更新。
//!
//! **2.8 位置编码**：自注意力本身是"无序"的（它对输入做的是加权求和，交换两个词的位置
//!   结果只是交换），所以必须额外注入位置信息。GPT-2 用的是**可学习的位置嵌入**，
//!   形状 `[context_length, emb_dim]`，而不是原始 Transformer 的正弦编码。
//!
//! 最终输入 = 词元嵌入 + 位置嵌入（逐元素相加，见书中图 2.17）。

use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::{Device, Int, Tensor, TensorData};

use super::dataset::Batch;

/// 词元嵌入 + 可学习位置嵌入。
///
/// 对应书中的：
/// ```python
/// self.tok_emb = nn.Embedding(vocab_size, emb_dim)
/// self.pos_emb = nn.Embedding(context_length, emb_dim)
/// ```
pub struct GptEmbedding {
    tok_emb: Embedding,
    pos_emb: Embedding,
    context_length: usize,
}

impl GptEmbedding {
    pub fn new(vocab_size: usize, emb_dim: usize, context_length: usize, device: &Device) -> Self {
        Self {
            // 默认初始化是 N(0, 1)——和 PyTorch 的 nn.Embedding 一致
            tok_emb: EmbeddingConfig::new(vocab_size, emb_dim).init(device),
            pos_emb: EmbeddingConfig::new(context_length, emb_dim).init(device),
            context_length,
        }
    }

    /// `ids: [batch, seq_len]`（整数） → `[batch, seq_len, emb_dim]`（浮点）
    pub fn forward(&self, ids: Tensor<2, Int>) -> Tensor<3> {
        let [batch, seq_len] = ids.dims();
        assert!(
            seq_len <= self.context_length,
            "序列长度 {seq_len} 超过上下文长度 {}",
            self.context_length
        );

        // 位置 ID：0..seq_len，然后广播到 batch 维（必须在 ids 被 move 之前算）
        // [seq_len] → [1, seq_len] → [1, seq_len, emb_dim] → [batch, seq_len, emb_dim]
        let device = ids.device();
        let positions = Tensor::<1, Int>::arange(0..seq_len as i64, &device).unsqueeze_dim::<2>(0);

        // [batch, seq_len] → [batch, seq_len, emb_dim]
        let token_embeddings = self.tok_emb.forward(ids);

        // positions 已经是 [1, seq_len]，Embedding 输出 [1, seq_len, emb_dim]，
        // 沿 batch 维复制即可与 token_embeddings 相加（广播加法不是必需的，因为形状已对齐）。
        let pos_embeddings = self.pos_emb.forward(positions).repeat_dim(0, batch);

        token_embeddings + pos_embeddings
    }

    /// 取出可学习的位置嵌入矩阵，用于观察"模型是否学会了位置的周期性结构"。
    pub fn position_weights(&self) -> Tensor<2> {
        self.pos_emb.weight.val()
    }
}

/// 把一个 batch 转成 Burn 张量对 `(inputs, targets)`。
///
/// 这就是书中 `DataLoader` 之后、`model(x)` 之前那一步。
pub fn batch_to_tensors(batch: &Batch, device: &Device) -> (Tensor<2, Int>, Tensor<2, Int>) {
    let (inputs, targets, shape) = batch.to_flat();
    (
        Tensor::<2, Int>::from_data(TensorData::new(inputs, shape), device),
        Tensor::<2, Int>::from_data(TensorData::new(targets, shape), device),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_shape_is_batch_seq_dim() {
        let device = Device::default();
        let emb = GptEmbedding::new(100, 16, 8, &device);
        let ids = Tensor::<2, Int>::from_data(TensorData::new(vec![1i64, 2, 3, 4], [1, 4]), &device);
        assert_eq!(emb.forward(ids).dims(), [1, 4, 16]);
    }

    #[test]
    fn position_embedding_makes_tokens_order_sensitive() {
        let device = Device::default();
        let emb = GptEmbedding::new(100, 8, 8, &device);

        // 同一批词元、不同顺序
        let a = Tensor::<2, Int>::from_data(TensorData::new(vec![1i64, 2], [1, 2]), &device);
        let b = Tensor::<2, Int>::from_data(TensorData::new(vec![2i64, 1], [1, 2]), &device);

        let out_a = emb.forward(a).to_data().try_to_vec::<f32>().unwrap();
        let out_b = emb.forward(b).to_data().try_to_vec::<f32>().unwrap();
        assert_ne!(out_a, out_b, "加入位置编码后，词序必须影响表示");
    }
}
