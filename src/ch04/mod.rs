//! 第 4 章：从零实现 GPT 模型并生成文本。
//!
//! 前四章的所有零件在这里汇合：第 2 章的嵌入、第 3 章的注意力、以及本章新增的
//! LayerNorm / GELU / 前馈网络 / 残差连接 / Pre-LN 组装。

pub mod attention;
pub mod feed_forward;
pub mod generate;
pub mod gpt;
pub mod layer_norm;
pub mod transformer_block;

pub use generate::{generate, sample_next, Sampling};
pub use gpt::{gpt_config_124m, GPTConfig, Gpt};
