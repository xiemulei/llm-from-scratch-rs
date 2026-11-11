//! 第 7 章：指令微调。
//!
//! 第 6 章的分类微调只输出 2 个类别；这一章要模型**生成完整的回答**。
//! 技术上就是继续做 next-token 预测，唯一的改动是：
//!
//! > 损失只在"回答"部分计算，指令部分全部掩掉。
//!
//! ```text
//! ### Instruction: ... ### Input: ... ### Response: 书被我读完了<|endoftext|>
//! └────────────── mask = false ──────────────┘  └── mask = true ──┘
//! ```

pub mod data;
pub mod train;
