//! 第 6 章演示：把 GPT-2 微调成垃圾短信分类器。
//!
//! 运行：`cargo run --release --bin ch06`
//!
//! 可选参数（没有 GPU 时用来控制耗时）：
//!   --epochs N        训练轮数（默认 1）
//!   --max-batches N   每轮最多用多少个训练批次（默认 40）
//!   --batch-size N    批大小（默认 8）
//!
//! 完整按书里的配置（5 个 epoch、170 个批次、batch 8）在 CPU 上要跑很久，
//! 所以这里默认只跑一个子集，但流程和指标定义完全一致。

use anyhow::Result;
use burn::tensor::Device;
use llm_from_scratch::ch02::gpt2_tokenizer::Gpt2Tokenizer;
use llm_from_scratch::ch04::gpt::{gpt_config_124m, Gpt};
use llm_from_scratch::ch05::weights::{load_gpt2_backbone, WeightFile};
use llm_from_scratch::ch06::classifier::SpamClassifier;
use llm_from_scratch::ch06::data::{
    balance, encode, label_counts, load_tsv, make_batches, shuffle, split, Batch, Example,
};
use llm_from_scratch::ch06::train::{classify, evaluate, fine_tune, ClassifyConfig};
use llm_from_scratch::section;

struct Args {
    epochs: usize,
    max_batches: usize,
    batch_size: usize,
    max_eval_batches: usize,
    eval_interval: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        epochs: 1,
        max_batches: 100,
        batch_size: 8,
        max_eval_batches: 15,
        eval_interval: 25,
    };
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i + 1 < args.len() {
        let v: usize = args[i + 1].parse().unwrap_or(0);
        match args[i].as_str() {
            "--epochs" => a.epochs = v.max(1),
            "--max-batches" => a.max_batches = v.max(1),
            "--batch-size" => a.batch_size = v.max(1),
            "--max-eval-batches" => a.max_eval_batches = v.max(1),
            "--eval-interval" => a.eval_interval = v.max(1),
            _ => {}
        }
        i += 2;
    }
    a
}

fn main() -> Result<()> {
    let args = parse_args();
    let device = Device::default();
    println!("设备: {device:?}");

    let tk = Gpt2Tokenizer::new()?;

    let (train_ex, val_ex, test_ex) = demo_6_2_data()?;
    let (train_b, val_b, test_b) = demo_6_3_batches(&train_ex, &val_ex, &test_ex, &tk, &device, &args);

    let model = demo_6_4_build(&device)?;

    demo_6_6_before(&model, &val_b);

    let model = demo_6_7_finetune(model, &train_b, &val_b, &args);

    demo_6_8_test(&model, &test_b, &tk, &device)?;

    Ok(())
}

/// 6.2 读取并切分数据集
fn demo_6_2_data() -> Result<(Vec<Example>, Vec<Example>, Vec<Example>)> {
    section("6.2 准备数据集");

    let mut data = load_tsv("data/sms_spam.tsv")?;
    println!("总样本: {}", data.len());
    let (ham, spam) = label_counts(&data);
    println!("  ham  {ham}（{:.1}%）", ham as f64 / data.len() as f64 * 100.0);
    println!("  spam {spam}（{:.1}%）", spam as f64 / data.len() as f64 * 100.0);
    println!("→ 严重不平衡：全猜 ham 就有 {:.1}% 准确率。", ham as f64 / data.len() as f64 * 100.0);

    shuffle(&mut data, 123);
    let (train, val, test) = split(data);
    let (h, s) = label_counts(&test);
    println!("\n切分后: train {} / val {} / test {}", train.len(), val.len(), test.len());
    println!("  测试集原始分布: ham {h}, spam {s}");

    // 测试集下采样成平衡集
    let test = balance(test);
    let (h, s) = label_counts(&test);
    println!("  下采样后: ham {h}, spam {s}（各占 50%）");

    Ok((train, val, test))
}

/// 6.3 编码 + 分批
fn demo_6_3_batches(
    train: &[Example],
    val: &[Example],
    test: &[Example],
    tk: &Gpt2Tokenizer,
    device: &Device,
    args: &Args,
) -> (Vec<Batch>, Vec<Batch>, Vec<Batch>) {
    section("6.3 编码与分批");

    let train_e = encode(train, tk);
    let val_e = encode(val, tk);
    let test_e = encode(test, tk);

    let lens: Vec<usize> = train_e.iter().map(|e| e.ids.len()).collect();
    let avg = lens.iter().sum::<usize>() as f64 / lens.len() as f64;
    let max = *lens.iter().max().unwrap();
    println!("短信长度（词元）: 平均 {avg:.1}，最长 {max}");

    let ad = device.clone().autodiff();
    let train_b = make_batches(&train_e, args.batch_size, &ad);
    let val_b = make_batches(&val_e, args.batch_size, &ad);
    let test_b = make_batches(&test_e, args.batch_size, &ad);

    println!(
        "批次: train {} / val {} / test {}（batch_size={}）",
        train_b.len(),
        val_b.len(),
        test_b.len(),
        args.batch_size
    );
    println!(
        "→ 同一 batch 内右侧用 <|endoftext|>(50256) 补齐到最长，不同 batch 长度可以不同。"
    );

    // CPU 上跑不动全量评估：训练和验证按参数截断，
    // 但测试集必须完整评估——平衡集里 spam 全排在前面，
    // 只取前几个批次会得到"全是 spam"的假指标。
    let n = train_b.len().min(args.max_batches);
    let v = val_b.len().min(args.max_eval_batches);
    println!(
        "实际使用: train {} 批、val {} 批、test {} 批（全量）",
        n,
        v,
        test_b.len()
    );
    (train_b[..n].to_vec(), val_b[..v].to_vec(), test_b)
}

/// 6.4 / 6.5 加载预训练权重 + 换输出头 + 冻结
fn demo_6_4_build(device: &Device) -> Result<SpamClassifier> {
    section("6.4-6.5 加载 GPT-2 并改造成分类器");

    let path = WeightFile::download("data")?;
    let weights = WeightFile::open(&path)?;

    let ad = device.clone().autodiff();
    let cfg = gpt_config_124m().with_tie_weights(false).with_drop_rate(0.1);
    let mut gpt = Gpt::new(&cfg, &ad);
    // load_gpt2 内部全程用 from_data 构造张量（叶子），所以可以直接在 autodiff 设备上加载
    load_gpt2_backbone(&weights, &mut gpt, &ad)?;
    println!("GPT-2 权重加载完成（输出头随后会被丢弃并替换）");

    let model = SpamClassifier::new(gpt, &ad);
    println!("替换输出头: 50257 → 2 类");
    println!("总参数量      : {}", burn::module::Module::num_params(&model));

    // 冻结：只训练最后一个 Transformer 块 + final_norm + 分类头
    let model = model.freeze_all_but_last(1);
    println!(
        "可训练参数量  : {}（只解冻最后 1 个 Transformer 块 + final_norm + 分类头）",
        model.trainable_params()
    );
    Ok(model)
}

/// 6.6 微调前的基线
fn demo_6_6_before(model: &SpamClassifier, val: &[Batch]) {
    section("6.6 微调前的基线");
    let m = evaluate(model, val);
    println!("验证集: loss {:.4}, 准确率 {:.3}", m.loss, m.accuracy);
    println!("→ 输出头是随机初始化的，准确率应接近 50%（二分类瞎猜）。");
}

/// 6.7 微调
fn demo_6_7_finetune(
    model: SpamClassifier,
    train: &[Batch],
    val: &[Batch],
    args: &Args,
) -> SpamClassifier {
    section("6.7 分类微调");
    let cfg = ClassifyConfig {
        epochs: args.epochs,
        eval_interval: args.eval_interval,
        ..Default::default()
    };
    let (model, _history) = fine_tune(model, train, val, &cfg);
    model
}

/// 6.8 在测试集上评估并使用模型
fn demo_6_8_test(
    model: &SpamClassifier,
    test: &[Batch],
    tk: &Gpt2Tokenizer,
    device: &Device,
) -> Result<()> {
    section("6.8 测试集评估与推理");

    let m = evaluate(model, test);
    println!("测试集（平衡集）: loss {:.4}, 准确率 {:.3}", m.loss, m.accuracy);

    println!("\n实际推理:");
    let samples = [
        ("Winner! You have won a $1000 gift card. Call now to claim!", 1),
        ("Hey, are we still meeting for lunch tomorrow?", 0),
        ("URGENT: your account will be suspended. Verify immediately.", 1),
        ("Ok, see you at the station at 6.", 0),
    ];

    for (text, truth) in samples {
        let ids = tk.encode(text);
        let n = ids.len();
        let batch = make_batches(
            &[llm_from_scratch::ch06::data::Encoded { ids, label: truth }],
            n.max(1),
            device,
        );
        let pred = classify(model, batch.into_iter().next().unwrap());
        let (label, conf) = pred[0];
        let name = if label == 1 { "spam" } else { "ham " };
        println!("  [{name}] {:.3}  {text}", conf);
    }

    println!("\n→ 只训练了最后一个 Transformer 块（约 7M 参数），");
    println!("  远少于全量微调的 124M，这就是「参数高效微调」的最简形式。");
    Ok(())
}
