//! 第 4 章演示：LayerNorm → GELU → 残差 → Transformer 块 → GPT → 文本生成。
//!
//! 运行：`cargo run --bin ch04`

use burn::module::Module;
use burn::tensor::{Device, Distribution, Int, Tensor, TensorData};
use llm_from_scratch::ch02::gpt2_tokenizer::Gpt2Tokenizer;
use llm_from_scratch::ch03::matrix_string;
use llm_from_scratch::ch04::{
    feed_forward::{gelu_tanh_manual, FeedForward},
    generate::{generate, Sampling},
    gpt::{gpt_config_124m, GPTConfig, Gpt},
    layer_norm::layer_norm_manual,
    transformer_block::TransformerBlock,
};
use llm_from_scratch::section;

fn main() {
    let device = Device::default();
    println!("设备: {device:?}");

    demo_4_2_layer_norm(&device);
    demo_4_3_gelu(&device);
    demo_4_5_block(&device);
    demo_4_6_gpt(&device);
    demo_4_7_generate(&device);
}

/// 4.2 LayerNorm
fn demo_4_2_layer_norm(device: &Device) {
    section("4.2 层归一化");
    let x = llm_from_scratch::ch03::matrix(
        vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![10.0, 20.0, 30.0, 40.0],
            vec![-5.0, 5.0, -5.0, 5.0],
        ],
        device,
    );
    println!("输入（注意各行尺度差异很大）:");
    println!("{}", matrix_string(&x));

    let d = 4;
    let gamma = Tensor::<2>::ones([1, d], device) * 1.0;
    let beta = Tensor::<2>::zeros([1, d], device);
    let out = layer_norm_manual(x.clone(), &gamma, &beta, 1e-5);
    println!("归一化后（每行均值 0、方差 1）:");
    println!("{}", matrix_string(&out));

    // 用 Burn 自带的 nn::LayerNorm 验证一致
    let mut ln = burn::nn::LayerNormConfig::new(d).init(device);
    ln.gamma = burn::module::Param::from_tensor(Tensor::ones([d], device));
    ln.beta = Some(burn::module::Param::from_tensor(Tensor::zeros([d], device)));
    let burn_out = ln.forward(x);
    let a = out.to_data().try_to_vec::<f32>().unwrap();
    let b = burn_out.to_data().try_to_vec::<f32>().unwrap();
    let max_diff = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    println!("手写实现与 Burn nn::LayerNorm 的最大误差: {max_diff:.2e}");
    println!("→ LayerNorm 沿特征维统计，与 batch 大小、序列长度无关。");
}

/// 4.3 GELU 前馈网络
fn demo_4_3_gelu(device: &Device) {
    section("4.3 GELU 激活与前馈网络");
    let xs: Vec<f32> = vec![-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
    let x = Tensor::<3>::from_data(
        TensorData::new(xs.clone(), [1, 1, xs.len()]),
        device,
    );
    let gelu = gelu_tanh_manual(x)
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();

    println!("  x      ReLU(x)   GELU(x)");
    for (i, &v) in xs.iter().enumerate() {
        println!("  {v:>5.1}   {:>7.3}   {:>7.3}", v.max(0.0), gelu[i]);
    }
    println!("→ 负值区 GELU 不为 0 而是一条平滑的尾巴，梯度更连续；");
    println!("  在 x≈-0.75 附近还有极小值，允许少量负激活通过。");

    let ff = FeedForward::new(768, device);
    let n = Module::num_params(&ff);
    println!("\n前馈网络 (768 → 3072 → 768) 参数量: {n}");
    println!("  计算: 768×3072 + 3072 + 3072×768 + 768 = {}", n);
}

/// 4.5 Transformer 块
fn demo_4_5_block(device: &Device) {
    section("4.4-4.5 残差连接与 Transformer 块（Pre-LN）");
    let cfg = GPTConfig::new(1000, 64, 64, 4, 1).with_drop_rate(0.0);
    let block = TransformerBlock::new(&cfg, device);
    let p = Module::num_params(&block);
    println!("单个 Transformer 块 (d=64) 参数量: {p}");

    let x = Tensor::<3>::random([1, 8, 64], Distribution::Default, device);
    let (out, weights) = block.forward_with_weights(x);
    println!("输入 [1, 8, 64] → 输出 {:?}", out.dims());
    println!("注意力权重形状: {:?}（batch, heads, seq, seq）", weights.dims());

    let w = weights
        .clone()
        .slice([0..1, 0..1])
        .squeeze_dim::<3>(1)
        .squeeze_dim::<2>(0);
    println!("\n第 0 个头的因果注意力（权重随机，仅看形状和掩码）:");
    println!("{}", matrix_string(&w));

    println!("Pre-LN 结构:  x = x + Attn(LayerNorm(x));  x = x + FFN(LayerNorm(x))");
    println!("→ 残差通路是一条纯恒等连接，梯度可直达底层，因此能堆很深。");
}

/// 4.6 完整 GPT 模型
fn demo_4_6_gpt(device: &Device) {
    section("4.6 组装 GPT 模型");

    // 先用小模型验证结构
    let small = GPTConfig::new(1000, 64, 64, 4, 2);
    let model = Gpt::new(&small, device);
    let idx = Tensor::<2, Int>::from_data(TensorData::new(vec![1i64, 2, 3, 4], [1, 4]), device);
    let logits = model.forward(idx);
    println!("小模型 [1, 4] 输入 → logits {:?}", logits.dims());
    println!("参数量: {}", Module::num_params(&model));

    // 书中 4.6 的 GPT_CONFIG_124M
    println!("\n--- GPT-2 small (124M) 配置 ---");
    let cfg = gpt_config_124m();
    println!(
        "vocab={} context={} emb={} heads={} layers={}",
        cfg.vocab_size, cfg.context_length, cfg.emb_dim, cfg.n_heads, cfg.n_layers
    );

    let model = Gpt::new(&cfg, device);
    let untied = model.logical_num_params();
    println!("参数量（不共享输出头）: {untied:>12}  ≈ {:.1}M", untied as f64 / 1e6);

    let model = Gpt::new(&cfg.clone().with_tie_weights(true), device);
    let tied = model.logical_num_params();
    println!("参数量（共享输出头）  : {tied:>12}  ≈ {:.1}M", tied as f64 / 1e6);
    println!(
        "→ 共享省下 {} 个参数（vocab×emb = {}×{}），占不共享版本的 {:.0}%",
        untied - tied,
        cfg.vocab_size,
        cfg.emb_dim,
        (untied - tied) as f64 / untied as f64 * 100.0
    );
    println!("  书中的 124M 指的是共享后的版本。");
    println!();
    println!("  注：Burn 的 Module::num_params() 按访问次数累加、不去重，");
    println!("  共享模式下它仍会报 163M，所以这里用 logical_num_params() 扣掉重复计数。");
}

/// 4.7 文本生成
fn demo_4_7_generate(device: &Device) {
    section("4.7 文本生成");

    // 生成演示用一个小模型（权重随机），重点是流程和三种解码策略的差异
    let cfg = GPTConfig::new(50257, 64, 128, 4, 2).with_drop_rate(0.0);
    let model = Gpt::new(&cfg, device);

    let tokenizer = Gpt2Tokenizer::new().ok();
    let prompt = "Every effort moves you";
    let ids: Vec<i64> = match &tokenizer {
        Some(tk) => tk.encode(prompt).iter().map(|&x| x as i64).collect(),
        None => vec![3626, 2177, 3621, 345],
    };
    let seq_len = ids.len();
    let idx = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, seq_len]), device);
    println!("提示词: {prompt:?} → {seq_len} 个词元");

    for sampling in [
        Sampling::Greedy,
        Sampling::Temperature(1.0),
        Sampling::TopK { k: 10, temperature: 1.0 },
    ] {
        let out = generate(&model, idx.clone(), 8, sampling, device);
        let out_ids: Vec<u32> = out
            .to_data()
            .try_to_vec::<i32>()
            .unwrap()
            .into_iter()
            .map(|x| x as u32)
            .collect();
        let text = match &tokenizer {
            Some(tk) => tk.decode(&out_ids),
            None => format!("{out_ids:?}"),
        };
        println!("\n{}:", sampling.describe());
        println!("  词元: {:?}", &out_ids[seq_len..]);
        println!("  文本: {text:?}");
    }

    println!("\n→ 权重全是随机初始化的，所以输出是乱码。");
    println!("  但三个关键点已经能看出来：");
    println!("  1. 贪心解码每次必然给出相同结果；");
    println!("  2. 温度采样每次不同；");
    println!("  3. 生成的序列越长，调用模型的次数越多（每步都要一次前向）。");
    println!("\n真正的文本要等第 5 章加载 GPT-2 官方权重，或自己预训练之后。");
}
