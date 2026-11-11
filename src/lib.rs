//! 《从零构建大模型》Rust / Burn 实现。
//!
//! 每一章一个模块，每个模块内部按书中节号组织（`2.2`、`2.3` ……），
//! 对应的可运行演示见 `src/bin/chXX.rs`。

pub mod ch01;
pub mod ch02;
pub mod ch03;
pub mod ch04;
pub mod ch05;
pub mod ch06;
pub mod ch07;

/// 统一的分隔线输出，让章节演示的打印结果像书里一样分块。
pub fn section(title: &str) {
    println!("\n{}", "=".repeat(72));
    println!("{title}");
    println!("{}", "=".repeat(72));
}
