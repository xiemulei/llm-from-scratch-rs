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

/// 选择执行设备并打印一行说明。
///
/// - 编译时只开了 `cpu`（默认）→ 落到 `Flex` 纯 Rust CPU 后端
/// - 编译时开了 `gpu` → `Device::default()` 会按「独显 > 集显 > CPU」自动选择，
///   也就是同一份代码在 Arc A770、RTX、Mac 上都能直接跑
/// - 环境变量 `BURN_DEVICE=wgpu|cpu|flex` 可以在运行时强制切换已编译的后端
pub fn init_device() -> burn::tensor::Device {
    let device = burn::tensor::Device::default();
    println!("设备: {device:?}");
    device
}

/// 统一的分隔线输出，让章节演示的打印结果像书里一样分块。
pub fn section(title: &str) {
    println!("\n{}", "=".repeat(72));
    println!("{title}");
    println!("{}", "=".repeat(72));
}
