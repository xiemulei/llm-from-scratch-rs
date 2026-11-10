# 从零构建大模型 · Rust / Burn 版

《Build a Large Language Model (From Scratch)》(Sebastian Raschka) 的 Rust 重写实现。
原书用 PyTorch，这里用 [Burn](https://burn.dev) 0.22 逐步复现，每一章一个可运行的二进制。

## 进度

| 章节 | 内容 | 状态 |
|---|---|---|
| 第 1 章 | 理解大语言模型（概念 / 全局图景） | ✅ |
| 第 2 章 | 文本数据处理：分词、BPE、滑动窗口、词嵌入 | ✅ |
| 第 3 章 | 注意力机制 | ✅ |
| 第 4 章 | 从零实现 GPT 模型 | ✅ |
| 第 5 章 | 在无标注数据上预训练 + 加载 GPT-2 权重 | ✅ |
| 第 6 章 | 文本分类微调（垃圾短信） | ✅ |
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
├── ch03/                 # 第 3 章：注意力机制
│   ├── simple.rs             3.3   简化自注意力（无参数）
│   ├── self_attention.rs     3.4   带 Wq/Wk/Wv 的自注意力
│   ├── causal.rs             3.5   因果掩码 + dropout
│   └── multi_head.rs         3.6   多头（朴素堆叠 vs 权重切分）
├── ch04/                 # 第 4 章：GPT 模型
│   ├── layer_norm.rs         4.2   手写 LayerNorm + 与 nn::LayerNorm 对照
│   ├── feed_forward.rs       4.3   GELU（tanh 近似）+ 前馈网络
│   ├── attention.rs          4.6   GPT-2 忠实版多头注意力（qkv_bias）
│   ├── transformer_block.rs  4.4-4.5 残差连接 + Pre-LN 组装
│   ├── gpt.rs                4.6   GPTConfig / GPTModel / 权重共享
│   └── generate.rs           4.7   贪心 / 温度 / Top-k 解码
├── ch05/                 # 第 5 章：预训练
│   ├── loss.rs               5.2   交叉熵 / 困惑度 / 错位
│   ├── text.rs               5.1   语料 → 词元 → 批次
│   ├── train.rs              5.3-5.4 训练循环 + 预热/余弦调度 + 梯度裁剪
│   └── weights.rs            5.5   加载 GPT-2 官方权重（safetensors）
├── ch06/                 # 第 6 章：分类微调
│   ├── data.rs               6.1-6.3 数据集切分 / 平衡 / 补齐 / 分批
│   ├── classifier.rs         6.4-6.6 换输出头 + 冻结 + last-token 分类
│   └── train.rs              6.7-6.8 微调循环与准确率评估
├── tools/
│   └── verify_numpy.py   # 独立的 numpy 参考实现，用来交叉验证推理结果
└── bin/
    ├── ch02.rs           # 逐节演示，对应书中的打印输出
    ├── ch03.rs
    ├── ch04.rs
    ├── ch05.rs
    └── ch06.rs
```

## 第 5 章的权重下载

`cargo run --release --bin ch05` 会自动下载 GPT-2 124M 权重到 `./data/gpt2.safetensors`（548MB）。
下载源按速度排序：**ModelScope**（实测 ~11MB/s）→ hf-mirror → 官方 HuggingFace。
支持断点续传，中断后重跑会从断点继续。

也可以手动下载后放到该路径：

```bash
curl -L -C - -o data/gpt2.safetensors   "https://modelscope.cn/api/v1/models/AI-ModelScope/gpt2/repo?Revision=master&FilePath=model.safetensors"
```

## 正确性验证

`tools/verify_numpy.py` 是一个只依赖 numpy 的独立 GPT-2 实现（直接读同一份 safetensors）。
两个实现的 logits 逐位一致，可以用于排查问题：

```bash
python tools/verify_numpy.py data/gpt2.safetensors
```

加载权重后对一句普通英文的困惑度是 **80.9**（perplexity），这就是"权重加载正确"的硬指标。

## 第 6 章：数据与控制耗时

垃圾短信数据集（UCI SMS Spam Collection，5574 条）已放在 `data/sms_spam.tsv`，
不需要联网下载。

CPU 上没有 GPU 时，可以限制训练规模：

```bash
cargo run --release --bin ch06 -- --epochs 1 --max-batches 100 --eval-interval 50
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--epochs` | 1 | 训练轮数（书中是 5） |
| `--max-batches` | 40 | 每轮最多用多少批次（书中是全部 488） |
| `--batch-size` | 8 | 批大小 |
| `--max-eval-batches` | 15 | 验证集最多评估多少批次 |
| `--eval-interval` | 25 | 每多少步评估一次 |

测试集**总是完整评估**：平衡集里 spam 全排在前面，只取前几批会得到假指标。
