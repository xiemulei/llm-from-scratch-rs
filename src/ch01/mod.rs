//! 第 1 章：理解大语言模型。
//!
//! 原书这一章是纯概念章节，没有可运行的代码。这里把关键结论整理成结构化的定义，
//! 后续每一章都会回头引用它们。

/// 大语言模型在书中的定义：一个被训练来"理解、生成、解释"自然语言的神经网络。
///
/// 全书只聚焦其中一类：**仅解码器的 Transformer（decoder-only transformer）**，
/// 也就是 GPT 系（GPT-2 / GPT-3 / ChatGPT 底座）的架构。
pub struct LlmDefinition;

/// 书里反复强调的一点：LLM 的训练分两个阶段，不是一个。
///
/// 这两个阶段对应第 5 章和第 6/7 章，是全书的结构骨架。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingStage {
    /// 阶段一：预训练（pretraining）。
    ///
    /// 在海量无标注文本上做"预测下一个词"（next-token prediction）。
    /// 目标函数是自监督的：文本本身就是标签，把每个位置的下一个词元当作 target。
    /// 产出的是**基座模型（base model）**，它擅长续写但不会听指令。
    Pretraining,
    /// 阶段二之一：指令微调 / 分类微调（finetuning）。
    ///
    /// 在有标注数据上继续训练，把基座模型改造成特定任务模型。
    /// 这一步数据量小几个数量级，但有监督。
    Finetuning,
}

/// 书中给出的 GPT-2 三个规格，用来理解"大模型"到底有多大。
///
/// 注意参数量主要由 `emb_dim² × n_layers` 决定，而计算量还额外正比于上下文长度。
#[derive(Debug, Clone, Copy)]
pub struct Gpt2Spec {
    pub name: &'static str,
    /// 隐藏层维度（d_model）
    pub emb_dim: usize,
    /// Transformer 块数量
    pub n_layers: usize,
    /// 注意力头数
    pub n_heads: usize,
    /// 上下文长度（最大序列长度）
    pub context_length: usize,
    /// 参数量（约）
    pub params: &'static str,
}

pub const GPT2_SMALL: Gpt2Spec = Gpt2Spec {
    name: "gpt2-small (124M)",
    emb_dim: 768,
    n_layers: 12,
    n_heads: 12,
    context_length: 1024,
    params: "124M",
};

pub const GPT2_MEDIUM: Gpt2Spec = Gpt2Spec {
    name: "gpt2-medium (355M)",
    emb_dim: 1024,
    n_layers: 24,
    n_heads: 16,
    context_length: 1024,
    params: "355M",
};

pub const GPT2_LARGE: Gpt2Spec = Gpt2Spec {
    name: "gpt2-large (774M)",
    emb_dim: 1280,
    n_layers: 36,
    n_heads: 20,
    context_length: 1024,
    params: "774M",
};

/// 全书的流程图：原始文本 → 分词 → 词元 ID → 嵌入 → Transformer → 输出概率。
///
/// 每一步对应一章：
/// 1. 分词与 BPE        → 第 2 章
/// 2. 注意力机制        → 第 3 章
/// 3. GPT 结构与生成    → 第 4 章
/// 4. 预训练            → 第 5 章
/// 5. 微调              → 第 6、7 章
pub fn pipeline() -> Vec<&'static str> {
    vec![
        "原始文本",
        "分词 (tokenizer) → 词元 ID",
        "词元嵌入 + 位置编码 → 向量序列",
        "N × Transformer 块（因果多头自注意力 + 前馈网络）",
        "输出层 → 词表大小的 logits",
        "softmax → 下一个词元的概率分布",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt2_small_matches_the_book() {
        assert_eq!(GPT2_SMALL.emb_dim, 768);
        assert_eq!(GPT2_SMALL.context_length, 1024);
        // 头数必须整除隐藏维度，否则多头拆分不成立
        assert_eq!(GPT2_SMALL.emb_dim % GPT2_SMALL.n_heads, 0);
    }

    #[test]
    fn every_stage_appears_in_the_pipeline() {
        assert_eq!(pipeline().len(), 6);
    }
}
