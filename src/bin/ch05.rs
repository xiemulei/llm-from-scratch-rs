//! 第 5 章演示：损失 → 训练 → 加载 GPT-2 官方权重。
//!
//! 运行：`cargo run --release --bin ch05`
//! 首次运行会下载约 550MB 的 GPT-2 权重到 `./data/`（走 ModelScope，约 1 分钟）。

use anyhow::Result;
use burn::module::Module;
use burn::tensor::{Device, Int, Tensor, TensorData};
use llm_from_scratch::ch02::gpt2_tokenizer::Gpt2Tokenizer;
use llm_from_scratch::ch04::generate::{generate, Sampling};
use llm_from_scratch::ch04::gpt::{GPTConfig, Gpt};
use llm_from_scratch::ch05::loss::{cross_entropy, perplexity};
use llm_from_scratch::ch05::text::{load_corpus, make_batches};
use llm_from_scratch::ch05::train::{evaluate, train, TrainConfig};
use llm_from_scratch::ch05::weights::{load_gpt2, WeightFile};
use llm_from_scratch::section;

fn main() -> Result<()> {
    let device = Device::default();
    println!("设备: {device:?}");

    let tk = Gpt2Tokenizer::new()?;
    demo_5_1_corpus(&tk);
    demo_5_2_loss(&device);
    demo_5_3_train(&device, &tk)?;
    demo_5_5_pretrained(&device, &tk)?;
    Ok(())
}

/// 5.1 语料与词元化
fn demo_5_1_corpus(tk: &Gpt2Tokenizer) {
    section("5.1 语料与词元化");
    let corpus = load_corpus("data/corpus.txt", 40).unwrap();
    let tokens = llm_from_scratch::ch05::text::encode(&corpus, tk);
    println!("语料字符数: {}", corpus.chars().count());
    println!("词元数    : {}", tokens.len());
    println!(
        "压缩比    : {:.2} 字符/词元",
        corpus.chars().count() as f64 / tokens.len() as f64
    );
    println!("前 10 个词元: {:?}", &tokens[..10.min(tokens.len())]);
    println!("解码还原  : {:?}", tk.decode(&tokens[..10.min(tokens.len())]));
}

/// 5.2 交叉熵与困惑度
fn demo_5_2_loss(device: &Device) {
    section("5.2 交叉熵损失与困惑度");

    let vocab = 50257;
    let uniform = (vocab as f32).ln();
    println!("随机初始化的模型:");
    println!("  损失   = ln({vocab}) = {uniform:.4}");
    println!("  困惑度 = {:.0}（相当于在 {vocab} 个候选之间犹豫）", perplexity(uniform));

    // 完美预测：目标位置给人造的高分
    let mut data = vec![0.0f32; 4 * 4];
    for i in 0..4 {
        data[i * 4 + 2] = 50.0;
    }
    let logits = Tensor::<3>::from_data(TensorData::new(data, [1, 4, 4]), device);
    let targets = Tensor::<2, Int>::from_data(TensorData::new(vec![2i64; 4], [1, 4]), device);
    let loss = cross_entropy(logits, targets).into_scalar::<f32>();
    println!("\n完美预测（目标 logit=50）:");
    println!("  损失 = {loss:.6} → 困惑度 = {:.4}", perplexity(loss));
    println!("→ 困惑度 = exp(损失)，可读作「模型平均在几个候选之间拿不定主意」。");
}

/// 5.3 / 5.4 训练循环 + 学习率调度
fn demo_5_3_train(device: &Device, tk: &Gpt2Tokenizer) -> Result<()> {
    section("5.3-5.4 训练循环");

    let corpus = load_corpus("data/corpus.txt", 40)?;
    let tokens = llm_from_scratch::ch05::text::encode(&corpus, tk);

    // 留出 10% 作验证集
    let split = tokens.len() * 9 / 10;
    let (train_tokens, val_tokens) = tokens.split_at(split);

    // autodiff 是设备上的装饰器：模型和批次都必须落在这个设备上
    let ad = device.clone().autodiff();

    let max_length = 64;
    let train_batches = make_batches(train_tokens, max_length, 32, 4, &ad);
    let val_batches = make_batches(val_tokens, max_length, 32, 4, &ad);
    println!(
        "训练批次 {} 个（每批 4×{max_length} 词元），验证批次 {} 个",
        train_batches.len(),
        val_batches.len()
    );

    // 124M 在 CPU 上训不动，用一个同构的小模型演示流程
    let cfg = GPTConfig::new(50257, 64, 64, 4, 2)
        .with_qkv_bias(true)
        .with_drop_rate(0.1);
    let model = Gpt::new(&cfg, &ad);
    println!("小模型参数量: {}", model.num_params());

    let before = evaluate(&model, &val_batches);
    println!("训练前验证损失: {before:.4}（困惑度 {:.0}）", perplexity(before));

    let train_cfg = TrainConfig {
        epochs: 3,
        eval_interval: 10,
        ..Default::default()
    };
    let (model, history) = train(model, &train_batches, &val_batches, &train_cfg);

    let after = evaluate(&model, &val_batches);
    println!(
        "训练后验证损失: {after:.4}（困惑度 {:.0}）→ 下降 {:.1}%",
        perplexity(after),
        (before - after) / before * 100.0
    );

    // 用训练好的小模型生成，看看有没有学到东西
    let prompt_ids: Vec<i64> = tk.encode("It is a truth").iter().map(|&x| x as i64).collect();
    let n = prompt_ids.len();
    let idx = Tensor::<2, Int>::from_data(TensorData::new(prompt_ids, [1, n]), device);
    let out = generate(&model, idx, 20, Sampling::TopK { k: 40, temperature: 0.8 }, device);
    let ids: Vec<u32> = out
        .to_data()
        .try_to_vec::<i32>()
        .unwrap()
        .into_iter()
        .map(|x| x as u32)
        .collect();
    println!("\n小模型生成（训练 3 个 epoch 后）:");
    println!("  {:?}", tk.decode(&ids));
    println!("→ 语料只有几千词元，能学到词形和局部搭配，但还谈不上通顺。");
    let _ = history;
    Ok(())
}

/// 5.5 加载 GPT-2 官方权重
fn demo_5_5_pretrained(device: &Device, tk: &Gpt2Tokenizer) -> Result<()> {
    section("5.5 加载 GPT-2 官方权重");

    let path = WeightFile::download("data")?;
    let weights = WeightFile::open(&path)?;
    println!(
        "权重文件: {} （{:.0} MB）",
        path.display(),
        std::fs::metadata(&path)?.len() as f64 / 1e6
    );

    let cfg = GPTConfig::new(50257, 1024, 768, 12, 12)
        .with_qkv_bias(true)
        .with_tie_weights(true)
        .with_drop_rate(0.0);
    let mut model = Gpt::new(&cfg, device);
    println!("参数量: {}（权重共享）", model.logical_num_params());

    load_gpt2(&weights, &mut model, device)?;
    println!("权重加载完成");

    // 困惑度校验：这是判断"权重加载是否正确"最硬的指标
    let probe = "It is a truth long known to engineers that a model can only learn what its data allows.";
    let probe_ids: Vec<i64> = tk.encode(probe).iter().map(|&x| x as i64).collect();
    let n = probe_ids.len();
    let probe_idx = Tensor::<2, Int>::from_data(TensorData::new(probe_ids, [1, n]), device);
    // 输入取前 n-1 个，目标取后 n-1 个
    let logits = model.forward(probe_idx.clone());
    let logits = logits.slice([0..1, 0..(n - 1), 0..50257]);
    let targets = probe_idx.slice([0..1, 1..n]);
    let loss = cross_entropy(logits, targets).into_scalar::<f32>();
    println!("
困惑度校验（GPT-2 应给出 30~100 的量级）:");
    println!("  文本: {probe:?}");
    println!("  损失 = {loss:.3} → 困惑度 = {:.1}", perplexity(loss));

    let prompts = [
        "Every effort moves you",
        "The sky is",
        "The capital of France is",
        "1, 2, 3, 4, 5,",
    ];

    for prompt in prompts {
        let ids: Vec<i64> = tk.encode(prompt).iter().map(|&x| x as i64).collect();
        let n = ids.len();
        let idx = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, n]), device);

        // 一次前向，看 top-5 预测
        let logits = model.forward(idx.clone());
        let last = logits
            .slice([0..1, (n - 1)..n, 0..50257])
            .squeeze_dim::<2>(1);
        let (top_vals, top_ids) = last.topk_with_indices(5, 1);
        let vals = top_vals.to_data().try_to_vec::<f32>().unwrap();
        let tids = top_ids.to_data().try_to_vec::<i32>().unwrap();

        println!("\n提示词: {prompt:?}");
        print!("  top-5 下一个词元: ");
        for (v, id) in vals.iter().zip(tids.iter()) {
            print!("{:?}({:.1}) ", tk.decode(&[*id as u32]), v);
        }
        println!();
    }

    for sampling in [
        Sampling::Greedy,
        Sampling::Temperature(0.7),
        Sampling::TopK { k: 40, temperature: 0.8 },
    ] {
        let ids: Vec<i64> = tk.encode("Every effort moves you").iter().map(|&x| x as i64).collect();
        let n = ids.len();
        let idx = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, n]), device);
        let out = generate(&model, idx, 40, sampling, device);
        let out_ids: Vec<u32> = out
            .to_data()
            .try_to_vec::<i32>()
            .unwrap()
            .into_iter()
            .map(|x| x as u32)
            .collect();
        println!("\n{}:", sampling.describe());
        println!("  {:?}", tk.decode(&out_ids));
    }

    println!("\n→ 这一章验证了前四章的每一行代码：");
    println!("  结构只要有一处和 GPT-2 不一致，加载权重时形状就会对不上，");
    println!("  或者生成出来的就不会是通顺英文。");
    Ok(())
}
