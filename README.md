# 从零构建大模型 · Rust / Burn 版

《Build a Large Language Model (From Scratch)》(Sebastian Raschka) 的 Rust 重写实现。
原书用 PyTorch，这里用 [Burn](https://burn.dev) 0.22 逐步复现，每一章一个可运行的二进制。

## 进度

| 章节 | 内容 | 状态 |
|---|---|---|
| 第 1 章 | 理解大语言模型（概念 / 全局图景） | ✅ |
| 第 2 章 | 文本数据处理：分词、BPE、滑动窗口、词嵌入 | ✅ |
| 第 3 章 | 注意力机制 | ⬜ |
| 第 4 章 | 从零实现 GPT 模型 | ⬜ |
| 第 5 章 | 在无标注数据上预训练 | ⬜ |
| 第 6 章 | 文本分类微调 | ⬜ |
| 第 7 章 | 指令跟随微调 | ⬜ |

## 运行

```bash
cargo run --bin ch02
```

## 关于 Burn 0.22

这个版本相比旧教程（0.13–0.18）有较大改动，写代码时注意：

- **没有 backend 泛型参数了**。`Tensor<2>` 只带维度；`Linear` / `Embedding` 等都不是泛型。
- `Device` 变成运行时枚举：`Device::cpu()`、`Device::wgpu(..)`、`Device::flex()`。
- 自动微分是装饰器：`let device = Device::default().autodiff();`
- NdArray 后端已废弃，纯 Rust CPU 后端改叫 `flex`。

好处是代码和 PyTorch 几乎一一对应，坏处是网上大部分 Burn 教程已经过时。

## 目录

```
src/
├── ch02/                 # 第 2 章：文本 → 数字
│   ├── simple_tokenizer.rs   2.2–2.4 简易分词与词表
│   ├── bpe.rs                2.5   从零实现字节对编码
│   ├── gpt2_tokenizer.rs     2.5   GPT-2 官方 BPE（tiktoken）
│   ├── dataset.rs            2.6   滑动窗口数据集
│   └── embeddings.rs         2.7–2.8 词元嵌入 + 位置编码
└── bin/
    └── ch02.rs           # 逐节演示，对应书中的打印输出
```
