//! 第 2 章：处理文本数据。
//!
//! 全书的目标是：把一段人话变成模型能算的浮点矩阵，再变回人话。
//! 本章只做前半段（文本 → 数字），路径是：
//!
//! ```text
//! 原始文本
//!   │  2.2 分词
//!   ▼
//! 词元列表 ["Hello", ",", " world"]
//!   │  2.3 / 2.4 词表映射
//!   ▼
//! 词元 ID  [15496, 11, 995]
//!   │  2.6 滑动窗口切片（input / target 错一位）
//!   ▼
//! 训练样本 (input_ids, target_ids)
//!   │  2.7 / 2.8 嵌入查表 + 位置编码
//!   ▼
//! [batch, seq_len, emb_dim] 浮点张量  ← 可以直接喂给 Transformer 了
//! ```

pub mod bpe;
pub mod dataset;
pub mod embeddings;
pub mod gpt2_tokenizer;
pub mod simple_tokenizer;

/// 演示用的小语料。
///
/// 原书用 Edith Wharton 的短篇《The Verdict》，这里换成一段自写文本，
/// 内容无关紧要，只是为了有几百个 token 跑通流程。
pub const SAMPLE_TEXT: &str = "\
It is a truth long known to engineers that a model can only learn what its data allows. \
The little machine sat on the desk and watched the words go by, one token at a time, \
and it learned nothing else but this: given a word, what comes next. \
Every morning the same story began again, and every evening the story ended in the same place. \
Some said the machine was foolish because it knew no facts at all. \
But it had read so much that the shape of language itself had settled into its weights, \
and when a question was asked, the answer came out sounding almost like a thought.";
