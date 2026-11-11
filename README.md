# 从零构建大模型 · Rust / Burn 版

《Build a Large Language Model (From Scratch)》(Sebastian Raschka) 的 Rust 重写实现。

原书用 PyTorch，这里用 [Burn](https://burn.dev) 0.22 逐步复现，**每一章一个可运行的二进制**，
每个模块内部按书中节号组织（`2.2`、`3.5` ……），测试也尽量对应书里的每个结论。

- 代码量：约 6000 行 Rust + 1 个独立的 numpy 参考实现
- 测试：85 个，`cargo test` 全绿（CPU / GPU 两种后端都验证过）
- 每一章都用一个独立的 git commit 提交

## 章节总览

| 章节 | 主题 | 本章产出 | 测试数 |
|---|---|---|---|
| [第 1 章](#第-1-章理解大语言模型) | 理解大语言模型 | 无代码，概念建模 | 2 |
| [第 2 章](#第-2-章处理文本数据) | 文本 → 数字 | 分词器 + BPE + 滑动窗口 + 嵌入 | 16 |
| [第 3 章](#第-3-章注意力机制) | 注意力机制 | 四个渐进版本 + 因果掩码 + 多头 | 15 |
| [第 4 章](#第-4-章从零实现-gpt-模型) | 从零实现 GPT | 完整 GPT-2 结构 + 文本生成 | 24 |
| [第 5 章](#第-5-章预训练与加载-gpt-2-权重) | 预训练 + 加载权重 | 训练循环 + 能说话的 GPT-2 | 13 |
| [第 6 章](#第-6-章文本分类微调) | 分类微调 | 垃圾短信分类器（78% 准确率） | 9 |
| [第 7 章](#第-7-章指令跟随微调) | 指令微调 | 带损失掩码的指令微调 | 9 |

---

## 环境要求

- Rust 1.85+（本项目在 1.99 上开发）
- 磁盘：`target/` 约 6GB，GPT-2 权重 548MB
- 内存：加载 124M 模型约需 2GB
- 网络：仅在下载 GPT-2 权重时需要（约 1 分钟）

## 快速开始

```bash
# 不需要任何 GPU，也不需要联网，立刻能跑
cargo run --release --bin ch02
cargo run --release --bin ch03
cargo run --release --bin ch04

# 第 5 章会下载 548MB 权重，然后让 GPT-2 说人话
cargo run --release --bin ch05

# 第 6、7 章会训练，默认只跑一个子集（几分钟）
cargo run --release --bin ch06
cargo run --release --bin ch07
```

首次编译约 4 分钟（依赖用 O3 编译，自己的代码用 O1 便于调试），之后都是秒级。

`--release` 是必须的：debug 模式下张量运算慢到无法接受。

---

## 后端选择：CPU / GPU 不用改代码

所有 bin 都调用 `llm_from_scratch::init_device()`，内部就是 `Device::default()`，
按「独显 > 集显 > CPU」自动选择，启动时打印实际选中项：

```
设备: Device<Cube(Wgpu(WgpuDevice { kind: DefaultDevice, backend: Auto }))>   ← GPU
设备: Device<Flex(Cpu)>                                                      ← CPU
```

| 命令 | 后端 | 适用 |
|---|---|---|
| `cargo run --release --bin chXX` | `Flex`（纯 Rust CPU） | 任何机器，开箱即用 |
| `cargo run --release --features gpu --bin chXX` | `Wgpu`（Vulkan/DX12/Metal/WebGPU） | NVIDIA / AMD / Intel / Apple 通用 |

运行时强制切换已编译的后端：`BURN_DEVICE=cpu` / `BURN_DEVICE=wgpu`。

> `gpu` feature 默认仍会同时编译 `cpu`。只想要 GPU 用
> `--no-default-features --features gpu`。

### 实测速度

```bash
cargo run --release --example bench_step                  # CPU
cargo run --release --features gpu --example bench_step   # GPU
```

GPT-2 124M 全参数训练步，batch 8 × seq 64，本机 Ryzen AI 9 H365 + Radeon 880M 核显：

| 后端 | 每步耗时 | 吞吐 | 等效算力 |
|---|---|---|---|
| Flex (CPU, 20 线程) | 4994 ms | 103 token/s | ~76 GFLOP/s |
| Wgpu (Radeon 880M 核显) | **791 ms** | **648 token/s** | **~482 GFLOP/s** |

核显上快 **6.3 倍**。Arc A770 / RTX 这类独显预期还能再快数倍。
首次运行 GPU 需要 JIT 编译 kernel（约 1 分钟），之后有缓存。

---

# 逐章详解

## 第 1 章：理解大语言模型

### 实现了什么

原书这一章是纯概念，没有可运行的代码。这里把关键结论整理成可引用的类型：

- `LlmDefinition` / `TrainingStage`：预训练与微调两个阶段
- `GPT2_SMALL` / `GPT2_MEDIUM` / `GPT2_LARGE`：三个规格的结构参数
- `pipeline()`：从原始文本到概率分布的六步流水线

### 重点

全书只聚焦 **decoder-only Transformer**。理解两条主线，后面每一章都是它们的展开：

1. **预训练**是自监督的：文本自己就是标签（预测下一个词元）
2. **微调**才是有监督的：分类用标签，指令微调用人工写的回答
3. 两个阶段的数据量差好几个数量级，但模型结构完全一样

### 难点

没有技术难点，难点是概念：弄清楚"基座模型"和"对话模型"的差别，
以及为什么"预测下一个词"这种简单目标能涌现出能力。

### 运行

```bash
cargo test ch01
```

---

## 第 2 章：处理文本数据

### 实现了什么

把一段人话变成模型能算的浮点矩阵。

| 小节 | 文件 | 内容 |
|---|---|---|
| 2.2 | `simple_tokenizer.rs` | 按空白与标点切分 |
| 2.3–2.4 | `simple_tokenizer.rs` | 词表双向映射 + `<\|unk\|>` 未登录词 |
| 2.5 | `bpe.rs` | **从零实现**字节对编码（可训练） |
| 2.5 | `gpt2_tokenizer.rs` | GPT-2 官方 50257 词表（tiktoken） |
| 2.6 | `dataset.rs` | 滑动窗口采样 + 确定性洗牌分批 |
| 2.7–2.8 | `embeddings.rs` | 词元嵌入 + 可学习位置编码 |

### 重点

- **自监督信号的来源**：`target` 恒等于 `input` 右移一位，这就是训练目标的全部秘密
- **词表法的死穴**：词表爆炸 + 未登录词 → 所以需要 BPE
- **byte-level BPE 的关键性质**：任何 UTF-8 文本都能无损往返，永不出现 `<unk>`
- **位置编码必须加**：自注意力本身是无序的（加权求和），交换位置只改变顺序

### 难点

1. **自己实现 BPE**：合并规则的频次统计、优先级排序、编码时的递归套用。
   书中用现成的 tiktoken，这里两种都给——手写版理解原理，tiktoken 版用于生产。
2. **Burn 0.22 的 `arange` 只接受 `&Device`**（实现了 `From<&Device>` 而非 `From<Device>`），
   写成 `device` 会编译失败。
3. **Rust 的 move 语义**：`Embedding::forward(ids)` 会消费掉 `ids`，
   位置编码必须在它之前算好。这是和 PyTorch 差异最大的地方。

### 运行

```bash
cargo run --release --bin ch02
```

逐节打印，其中 GPT-2 分词结果和书里完全一致：

```
"Hello, do you like tea?" → [15496, 11, 466, 345, 588, 8887, 30]
```

网速测试（验证指纹）：

```bash
cargo test --lib ch02 -- --ignored   # 需要联网下载 GPT-2 词表
```

本章共 16 个测试，其中 3 个标记为 `#[ignore]`（依赖网络下载词表）。

---

## 第 3 章：注意力机制

### 实现了什么

四个渐进版本，每一步都能单独跑、单独验证。

| 小节 | 文件 | 版本 | 可学习参数 | 因果 |
|---|---|---|---|---|
| 3.3 | `simple.rs` | 简化自注意力 | ❌ | ❌ |
| 3.4 | `self_attention.rs` | 带 Wq/Wk/Wv | ✅ | ❌ |
| 3.5 | `causal.rs` | 因果掩码 + dropout | ✅ | ✅ |
| 3.6 | `multi_head.rs` | 多头（两种实现） | ✅ | ✅ |

### 重点

- **Q/K/V 的角色**：Q 是"我在找什么"，K 是"我是什么"，V 是"我能提供什么"
- **为什么要除以 √d_k**：点积的方差随维度线性增长，不缩放的话 softmax 会退化成 one-hot
- **因果掩码用 `-∞` 而不是 `0`**：`exp(0) = 1` 仍然贡献权重，必须 `-∞` 才能彻底消失
- **多头的两种实现参数量差 `n_heads` 倍**：朴素堆叠每个头都投影到完整 `d_out`，
  而权重切分每个头只投影到 `head_dim`

### 难点

1. **Burn 的 `tril_mask` 语义和 PyTorch 相反**。这是本章最大的坑：

   ```rust
   // Burn：名字叫 mask，语义是"待填充的位置" → 上三角为 true
   Tensor::<2, Bool>::tril_mask([3, 3], 0, &device)
   // [[false,  true,  true ],
   //  [false, false, true ],
   //  [false, false, false]]

   // PyTorch：torch.tril 返回下三角为 true
   ```

   我先写了 `.bool_not()`，结果未来反而被保留、过去被屏蔽，最后一行全 `-inf` 出 NaN。

2. **`softmax` 从方法变成了自由函数**：0.22 里没有 `tensor.softmax(dim)`，
   得写 `burn::tensor::activation::softmax(t, dim)`。

3. **`Dropout` 只在 autodiff 设备上生效**。源码里明确判断
   `if !input.device().is_autodiff() { return input; }`，
   所以纯推理设备上它是恒等映射——第 5 章真正训练时必须记得这点。

### 运行

```bash
cargo run --release --bin ch03
```

会打印 ASCII 热力图，因果掩码的效果一眼可见（右上角恒为空白）：

```
  @
  = +
  - - -
  : : : :
  : : : : :
  . . . : . :
```

以及初始化差异的实测对比（书中 3.4 的结论）：

| 初始化 | 注意力最大行内权重（均匀应为 0.1667） |
|---|---|
| v1 `U(0,1)` | **0.2577** — 尖锐但和语义无关 |
| v2 Kaiming（`nn.Linear` 默认） | **0.1835** — 接近均匀，训练后才会分化 |

---

## 第 4 章：从零实现 GPT 模型

### 实现了什么

把前三章的零件拼成完整的 GPT-2，并让它生成文本。

| 小节 | 文件 | 内容 |
|---|---|---|
| 4.2 | `layer_norm.rs` | 手写 LayerNorm + 与 `nn::LayerNorm` 逐元素比对 |
| 4.3 | `feed_forward.rs` | GELU（tanh 近似）+ 前馈网络 |
| 4.4–4.5 | `transformer_block.rs` | 残差连接 + Pre-LN 组装 |
| 4.6 | `attention.rs` | GPT-2 忠实版多头注意力（`qkv_bias`） |
| 4.6 | `gpt.rs` | `GPTConfig` / `GPTModel` / 权重共享 |
| 4.7 | `generate.rs` | 贪心 / 温度 / Top-k 解码 |

### 重点

- **LayerNorm vs BatchNorm**：前者在**特征维**统计，与 batch 无关，序列长度可任意变化
- **GELU 的负值尾巴**：`x=-1` 处 GELU ≈ -0.159 而 ReLU 是 0，梯度更连续
- **Pre-LN vs Post-LN**：`x = x + f(LN(x))` 让残差通路是一条纯恒等连接，
  梯度可直达底层，才能堆得深
- **权重共享**：`tok_emb` 和 `out_head` 是同一个矩阵，省掉 38.6M 参数（占 24%）

### 难点

1. **`Linear` 的权重布局和 PyTorch 相反**（本章最大的坑）：

   ```
   Burn     : weight.shape = [d_input, d_output]   ← 行主序
   PyTorch  : weight.shape = [d_output, d_input]
   ```

   有意思的是这反而和 GPT-2 的 `Conv1D` 一致，第 5 章加载权重时省掉了转置。

2. **`Param::clone()` 不是梯度级共享**。源码注释明确写着
   *"Any mutation forks into a new LazyInitState"*，所以优化器一更新参数两个句柄就分叉。
   真正的权重共享要在 `forward` 里直接用 `tok_emb.weight` 算 logits。

3. **`Module::num_params()` 不去重**：同一 Param 被访问两次就算两遍，
   共享模式下仍报 163M，所以另外写了 `logical_num_params()`。

4. **`argmax` 保留维度**：返回 `[b, 1]` 而不是 `[b]`（和 numpy 不同）。

### 运行

```bash
cargo run --release --bin ch04
```

参数量和原书完全对上：

```
参数量（不共享输出头）: 163,037,184
参数量（共享输出头）  : 124,439,808   ← 和书里印的一致
```

生成的文本是乱码（权重随机），但结构已经完整：

```
"Every effort moves you Somethingfreefree Editor outlets aber outlets Oy"
```

---

## 第 5 章：预训练与加载 GPT-2 权重

### 实现了什么

**本章是分水岭**：前四章的结构只要能加载官方权重并说出人话，就证明实现是对的。

| 小节 | 文件 | 内容 |
|---|---|---|
| 5.1 | `text.rs` | 语料 → 词元 → 滑动窗口批次 |
| 5.2 | `loss.rs` | 交叉熵 / 困惑度 / next-token 错位 |
| 5.3–5.4 | `train.rs` | AdamW + 线性预热 + 余弦退火 + 梯度裁剪 |
| 5.5 | `weights.rs` | 下载并加载 GPT-2 官方权重（safetensors） |

### 重点

- **困惑度 = exp(损失)**，可读作"模型平均在几个候选之间拿不定主意"
- **预热 + 余弦退火**：Adam 的动量估计在最初几步不可靠，直接从大学习率开始容易炸
- **权重共享时存 124M，不共享时 163M**，书中的"124M"指共享后的版本

### 难点

1. **权重下载**。550MB 的文件，实测：

   | 源 | 速度 | 耗时 |
   |---|---|---|
   | 官方 HuggingFace | ~100 KB/s | ~90 分钟 |
   | hf-mirror | 302 到 CDN，不稳定 | 经常断 |
   | **ModelScope** | **~11 MB/s** | **~1 分钟** |

   默认走 ModelScope，带 `Range` 断点续传（实测第一次下到 209MB 就断了，续传接上）。

2. **键名映射 + QKV 三合一**。GPT-2 把 query/key/value 合并成一个 `c_attn` `[768, 2304]`，
   必须沿第 1 维切成三份。因为 Burn 的 `Linear` 布局恰好和 GPT-2 的 `Conv1D` 一致，
   除 `lm_head` 外全部可以原样搬运。

3. **autodiff 设备不接受非叶子张量**。用 `tensor.slice()` 切 QKV 会创建带计算图节点的
   中间张量，`Param::from_tensor` 直接 panic。改成从原始 f32 数据重新构造（天然是叶子）。

### 运行

```bash
cargo run --release --bin ch05
```

输出（这是全书最爽的一刻）：

```
困惑度校验（GPT-2 应给出 30~100 的量级）:
  损失 = 4.394 → 困惑度 = 80.9

提示词: "Every effort moves you"
  top-5: " forward"(-131.9) " to"(-132.4) " along"(-133.1) "."(-133.2) ","(-133.2)

贪心解码:
  "Every effort moves you forward.\n\nThe first step is to understand the importance of your work."

温度采样 (T=0.7):
  "Every effort moves you to keep it moving. It's not easy. I'm not saying that you can't do it..."
```

自训练部分（6.5M 小模型，3 个 epoch）：

```
训练前验证损失: 10.8537（困惑度 51725）
训练后验证损失:  8.7658（困惑度  6412）→ 下降 19.2%
```

### 数据准备

`data/gpt2.safetensors`（548MB）不存在时会自动下载。也可以手动：

```bash
curl -L -C - -o data/gpt2.safetensors \
  "https://modelscope.cn/api/v1/models/AI-ModelScope/gpt2/repo?Revision=master&FilePath=model.safetensors"
```

### 怎么确认实现是对的

三个层次的验证，缺一不可：

1. **权重本身**：`wte` 前三行数值与书中打印的 `params["wte"]` 逐个吻合
2. **独立实现交叉验证**：`tools/verify_numpy.py` 是一个只依赖 numpy 的 GPT-2 实现，
   和 Rust 版 logits **逐位一致**

   ```bash
   python tools/verify_numpy.py data/gpt2.safetensors
   ```

3. **困惑度**：一句普通英文必须是 **80.9**。这是"权重加载正确"的硬指标

> ⚠️ 踩坑记录：我曾用手抄的 token id 做诊断，得到困惑度 4224 的假警报，
> 排查了两个小时。**任何数值诊断都必须用真正的分词器输出**。

---

## 第 6 章：文本分类微调

### 实现了什么

把只会"续写"的 GPT-2 改造成会"判断"的分类器。

| 小节 | 文件 | 内容 |
|---|---|---|
| 6.1–6.3 | `data.rs` | 数据集切分 / 平衡 / 补齐 / 分批 |
| 6.4–6.6 | `classifier.rs` | 换输出头 + 冻结 + last-token 分类 |
| 6.7–6.8 | `train.rs` | 微调循环与准确率评估 |

### 重点

三处改动：

1. **换输出头**：`768 → 50257` 换成 `768 → 2`
2. **取最后一个 token 的隐藏状态**：因果模型里它就是整句的"摘要"
3. **冻结大部分参数**：只训练最后 1 个 Transformer 块 + final_norm + 分类头

```
总参数量      : 163,038,722
可训练参数量  :   7,090,946   ← 只占 4.3%
```

冻结的三条理由：数据量小（3901 条）会过拟合、训练参数少一个数量级、
GPT-2 低层学到的是通用语言特征本来就不该动。这就是"参数高效微调"的最简形式。

### 难点

1. **`to_device` 搬到 autodiff 设备会让冻结失效**。一开始"普通设备建模型 → 加载 →
   to_device(autodiff)"，结果可训练参数只剩分类头的 **1538** 个。
   最终改为直接在 autodiff 设备上构建和加载。

2. **平衡测试集必须整体评估**。`balance()` 把 spam 排在前面，
   只取前 15 个批次等于全在 spam 上评估，准确率会显示成 **0.042** 这种荒谬数字。

3. **验证集本身是 86% ham 的不平衡集**，所以"验证准确率 0.86"很可能只是学到了
   全猜 ham——必须看测试集的平衡集结果才有意义。

### 运行

```bash
cargo run --release --bin ch06                       # 默认 100 步，约 3 分钟
cargo run --release --bin ch06 -- --epochs 1 --max-batches 500   # 完整 1 轮，约 10 分钟
cargo run --release --features gpu --bin ch06        # 有 GPU 快 6 倍
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--epochs` | 1 | 训练轮数（书中是 5） |
| `--max-batches` | 100 | 每轮最多用多少批次（全量 488） |
| `--batch-size` | 8 | 批大小 |
| `--max-eval-batches` | 15 | 验证集最多评估多少批次 |
| `--eval-interval` | 25 | 每多少步评估一次 |

实测结果（完整 1 轮 = 488 步，约 10 分钟）：

```
验证集 loss  1.2241 → 0.2948
测试集（平衡集）准确率  0.781      ← 随机基线 0.5
```

书中 5 个 epoch 能到 0.97 左右；CPU 上跑 5 轮要 50 分钟，所以默认只跑子集。

---

## 第 7 章：指令跟随微调

### 实现了什么

让模型学会**按指令生成回答**，而不是只判断类别。

| 小节 | 文件 | 内容 |
|---|---|---|
| 7.1–7.4 | `data.rs` | Alpaca 模板 / 损失掩码 / 分批 |
| 7.5–7.7 | `train.rs` | 掩码交叉熵 / 微调 / 生成回答 |

### 重点

技术上仍然是 next-token 预测，**唯一的新概念是损失掩码**：

```
### Instruction: ... ### Input: ... ### Response: <回答><|endoftext|>
└──────────── mask = false ────────────┘   └── mask = true ──┘
```

模型该学的是"怎么答"，而不是"怎么复述题目"。所以指令区的掩码全为 false，
`masked_cross_entropy` 只对 true 的位置求平均。

书中用 355M 的 gpt2-medium；CPU 上跑不动，这里用 124M，流程完全一致（不冻结，全量微调）。

### 难点

1. **`ignore_index = -100` 和 Burn 的 `gather` 不兼容**。书里把无效位置的 target 设成 -100，
   但 `gather` 不接受非法下标。
   **解决**：target 始终合法，忽略逻辑完全交给 mask 张量——语义等价，数据上更安全。

2. **目标错位时要连掩码一起左移**：`target[i] = input[i+1]`，`mask[i] = mask_orig[i+1]`，
   否则会把回答的第一个词元漏掉。

3. **模板细节**：有 `input` 和没 `input` 是两套前缀（书中 7.2 的做法），
   `format_input` 里分支处理。

### 运行

```bash
cargo run --release --bin ch07                     # 默认 60 步，约 6 分钟
cargo run --release --bin ch07 -- --epochs 5 --max-batches 117   # 按书里的配置
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--epochs` | 1 | 训练轮数（书中是 5） |
| `--max-batches` | 60 | 每轮最多用多少批次（全量 117） |
| `--batch-size` | 8 | 批大小 |

实测（60 步，约 6 分钟）：

```
微调前验证损失: 3.8627
微调后验证损失: 1.6835

指令: Identify the correct spelling of the word.
输入: receeve
回答: "The correct spelling of the word is 'receeve.'"
```

抓住了模板和句式，但还没真正纠错——如实说明：书中用 355M + 5 epoch 才有像样的效果。

---

# 附录

## 目录结构

```
src/
├── ch01/mod.rs                 # 概念建模
├── ch02/                       # 第 2 章：文本 → 数字
│   ├── simple_tokenizer.rs     2.2–2.4 简易分词与词表
│   ├── bpe.rs                  2.5     从零实现字节对编码
│   ├── gpt2_tokenizer.rs       2.5     GPT-2 官方 BPE（tiktoken）
│   ├── dataset.rs              2.6     滑动窗口数据集
│   └── embeddings.rs           2.7–2.8 词元嵌入 + 位置编码
├── ch03/                       # 第 3 章：注意力
│   ├── simple.rs               3.3     简化自注意力（无参数）
│   ├── self_attention.rs       3.4     带 Wq/Wk/Wv
│   ├── causal.rs               3.5     因果掩码 + dropout
│   └── multi_head.rs           3.6     多头（朴素堆叠 vs 权重切分）
├── ch04/                       # 第 4 章：GPT 模型
│   ├── layer_norm.rs           4.2     手写 LayerNorm + 与内置对照
│   ├── feed_forward.rs         4.3     GELU + 前馈网络
│   ├── attention.rs            4.6     GPT-2 忠实版（qkv_bias）
│   ├── transformer_block.rs    4.4-4.5 残差 + Pre-LN
│   ├── gpt.rs                  4.6     GPTConfig / GPTModel / 权重共享
│   └── generate.rs             4.7     贪心 / 温度 / Top-k
├── ch05/                       # 第 5 章：预训练
│   ├── loss.rs                 5.2     交叉熵 / 困惑度
│   ├── text.rs                 5.1     语料 → 批次
│   ├── train.rs                5.3-5.4 训练循环 + 学习率调度
│   └── weights.rs              5.5     加载 GPT-2 权重
├── ch06/                       # 第 6 章：分类微调
│   ├── data.rs                 6.1-6.3 数据集处理
│   ├── classifier.rs           6.4-6.6 分类器 + 冻结
│   └── train.rs                6.7-6.8 微调与评估
├── ch07/                       # 第 7 章：指令微调
│   ├── data.rs                 7.1-7.4 模板 + 损失掩码
│   └── train.rs                7.5-7.7 掩码损失 + 生成
├── lib.rs                      # init_device() / section()
└── bin/ch01..ch07.rs           # 每章一个可运行演示

examples/
├── bench_step.rs               # 训练步速 benchmark（CPU/GPU 对比）
└── dump_ids.rs                 # 打印任意文本的 GPT-2 词元 ID

tools/
└── verify_numpy.py             # 独立的 numpy 参考实现

data/
├── sms_spam.tsv                # 5574 条垃圾短信（已入库）
├── instruction-data.json       # 1100 条指令数据（已入库）
└── gpt2.safetensors            # 548MB，自动下载，不入库
```

## Burn 0.22 API 备忘

这个版本相比旧教程（0.13–0.18）有大改动，网上大部分教程已经过时：

| 旧写法 | 0.22 实际写法 |
|---|---|
| `Tensor<B, 2>` | `Tensor<2>` |
| `struct Model<B: Backend> { .. }` | `struct Model { .. }` |
| `NdArrayDevice::Cpu` | `Device::default()` / `Device::cpu()` |
| `Autodiff<NdArray>` 类型 | `device.autodiff()` 装饰器 |
| `NdArray` 后端 | 已 `#[deprecated]`，改用 `flex` |
| `tensor.softmax(dim)` | `activation::softmax(t, dim)` |

### 踩坑合集

| 现象 | 原因 | 解法 |
|---|---|---|
| `Unsqueeze` rank 错误 | `Embedding` 输出已是 3 维还再 unsqueeze | 直接 `repeat_dim` 广播 |
| 因果掩码方向反了 | Burn `tril_mask` 语义与 `torch.tril` 相反 | **不要** `bool_not()` |
| 最后一行为 NaN | 该行被全部掩成 `-inf` | 检查掩码方向 |
| `DTypeMismatch: Bool(Native) vs Bool(U32)` | Bool 张量布局后端相关 | 先 `.int()` 再读 `i32` |
| `Can't convert a non leaf tensor` | `slice`/`swap_dims` 产生中间张量 | `into_data()` 后重建 |
| 冻结失效，可训练参数只剩分类头 | 用 `to_device` 搬到 autodiff 设备 | 直接在 autodiff 设备上构建 |
| `num_params` 比预期多 | 访问次数累加，不去重 | 自己扣掉重复计数 |

## 与原书的差异

诚实记录，方便对照：

| 项目 | 原书 | 本项目 | 原因 |
|---|---|---|---|
| 第 5 章训练语料 | 《The Verdict》 | 自写英文（重复） | 版权 |
| 第 6 章训练轮数 | 5 epoch | 默认 1 epoch 的子集 | CPU 太慢 |
| 第 7 章模型 | gpt2-medium (355M) | gpt2-small (124M) | CPU 太慢 |
| 第 2 章 BPE | 直接用 tiktoken | 手写 + tiktoken 两套 | 教学价值 |
| 第 7 章损失掩码 | `ignore_index = -100` | 独立 mask 张量 | Burn 的 gather 限制 |

## 已知限制

- **没实现 KV cache**：每次生成都重算整个前缀，长文本生成慢。书中同样没有。
- **没实现 LoRA**：书末附录 E 的内容。
- **没有分布式**：`burn-distributed` 尚在早期。
- **CPU 上训练 124M 很慢**：先用 GPU 特性，或者按上面的参数跑子集。

## 参考

- 原书：[Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- 原书代码：https://github.com/rasbt/LLMs-from-scratch
- Burn 文档：https://burn.dev/books/burn/
- Burn 仓库：https://github.com/tracel-ai/burn
