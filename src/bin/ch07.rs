//! 第 7 章演示：指令微调 GPT-2。
//!
//! 运行：`cargo run --release --bin ch07 -- --epochs 1 --max-batches 100`
//!
//! 参数：
//!   --epochs N        训练轮数（默认 1，书中是 5）
//!   --max-batches N   每轮最多用多少训练批次（默认 100，全书 1100 条约 138 批）
//!   --batch-size N    批大小（默认 8）

use anyhow::Result;
use burn::tensor::Device;
use llm_from_scratch::ch02::gpt2_tokenizer::Gpt2Tokenizer;
use llm_from_scratch::ch04::gpt::{gpt_config_124m, Gpt};
use llm_from_scratch::ch05::weights::{load_gpt2, WeightFile};
use llm_from_scratch::ch07::data::{encode, format_input, load_json, make_batches, InstructionExample};
use llm_from_scratch::ch07::train::{fine_tune, generate_response, InstructConfig};
use llm_from_scratch::section;

struct Args {
    epochs: usize,
    max_batches: usize,
    batch_size: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        epochs: 1,
        max_batches: 60,
        batch_size: 8,
    };
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i + 1 < args.len() {
        let v: usize = args[i + 1].parse().unwrap_or(0);
        match args[i].as_str() {
            "--epochs" => a.epochs = v.max(1),
            "--max-batches" => a.max_batches = v.max(1),
            "--batch-size" => a.batch_size = v.max(1),
            _ => {}
        }
        i += 2;
    }
    a
}

fn main() -> Result<()> {
    let args = parse_args();
    let device = llm_from_scratch::init_device();

    let tk = Gpt2Tokenizer::new()?;

    let data = demo_7_2_data()?;
    let (train_b, val_b) = demo_7_3_batches(&data, &tk, &device, &args);

    let model = demo_7_5_load(&device)?;

    let model = demo_7_6_train(model, &train_b, &val_b, &args);

    demo_7_7_generate(&model, &data, &tk, &device);

    Ok(())
}

/// 7.2 数据集
fn demo_7_2_data() -> Result<Vec<InstructionExample>> {
    section("7.2 指令数据集");
    let data = load_json("data/instruction-data.json")?;
    let with_input = data.iter().filter(|e| !e.input.trim().is_empty()).count();
    println!("样本数: {}", data.len());
    println!("其中带 input 的: {with_input}，纯指令的: {}", data.len() - with_input);

    println!("\n模板示例:");
    println!("{}", format_input(&data[0]).replace('\n', "\n  "));
    println!("  {}", data[0].output);

    let (train_n, val_n) = (data.len() * 85 / 100, data.len());
    println!("\n切分: train [0, {train_n}) / val [{train_n}, {val_n})");
    Ok(data)
}

/// 7.3 / 7.4 编码与分批（重点看掩码）
fn demo_7_3_batches(
    data: &[InstructionExample],
    tk: &Gpt2Tokenizer,
    device: &Device,
    args: &Args,
) -> (Vec<llm_from_scratch::ch07::data::InstructionBatch>, Vec<llm_from_scratch::ch07::data::InstructionBatch>) {
    section("7.3-7.4 编码与掩码");

    let split = data.len() * 85 / 100;
    let train_e: Vec<_> = data[..split].iter().map(|e| encode(e, tk)).collect();
    let val_e: Vec<_> = data[split..].iter().map(|e| encode(e, tk)).collect();

    // 展示第一条样本的掩码
    let first = &train_e[0];
    let first_true = first.mask.iter().position(|&b| b).unwrap_or(0);
    println!("第一条样本: 共 {} 个词元", first.ids.len());
    println!("  指令部分 {} 个词元（mask=false）", first_true);
    println!("  回答部分 {} 个词元（mask=true）", first.ids.len() - first_true);
    println!("  解码回答: {:?}", tk.decode(&first.ids[first_true..]));
    println!("→ 损失只在 mask=true 的位置计算，模型学的是「怎么答」而不是「怎么抄题」。");

    let ad = device.clone().autodiff();
    let train_b = make_batches(&train_e, args.batch_size, &ad);
    let val_b = make_batches(&val_e, args.batch_size, &ad);
    println!(
        "\n批次: train {} / val {}（batch_size={}）",
        train_b.len(),
        val_b.len(),
        args.batch_size
    );

    let n = train_b.len().min(args.max_batches);
    let v = val_b.len().min(10);
    (train_b[..n].to_vec(), val_b[..v].to_vec())
}

/// 7.5 加载预训练权重
fn demo_7_5_load(device: &Device) -> Result<Gpt> {
    section("7.5 加载预训练 GPT-2");
    let path = WeightFile::download("data")?;
    let weights = WeightFile::open(&path)?;

    let ad = device.clone().autodiff();
    // 书中用 355M 的 gpt2-medium；CPU 上跑不动，这里用 124M，流程一致
    let cfg = gpt_config_124m().with_tie_weights(true).with_drop_rate(0.1);
    let mut model = Gpt::new(&cfg, &ad);
    load_gpt2(&weights, &mut model, &ad)?;
    println!("参数量: {}（gpt2-small；书中用 355M 的 medium）", model.logical_num_params());
    Ok(model)
}

/// 7.6 微调
fn demo_7_6_train(
    model: Gpt,
    train: &[llm_from_scratch::ch07::data::InstructionBatch],
    val: &[llm_from_scratch::ch07::data::InstructionBatch],
    args: &Args,
) -> Gpt {
    section("7.6 指令微调");
    let before = llm_from_scratch::ch07::train::evaluate(&model, val);
    println!("微调前验证损失: {before:.4}");

    let cfg = InstructConfig {
        epochs: args.epochs,
        eval_interval: 20,
        ..Default::default()
    };
    let (model, _h) = fine_tune(model, train, val, &cfg);

    let after = llm_from_scratch::ch07::train::evaluate(&model, val);
    println!("微调后验证损失: {after:.4}");
    model
}

/// 7.7 生成回答
fn demo_7_7_generate(model: &Gpt, data: &[InstructionExample], tk: &Gpt2Tokenizer, device: &Device) {
    section("7.7 生成回答");

    // 用训练集之外的两条样本
    let samples = [
        InstructionExample {
            instruction: "Identify the correct spelling of the word.".into(),
            input: "receeve".into(),
            output: String::new(),
        },
        InstructionExample {
            instruction: "Rewrite the sentence in passive voice.".into(),
            input: "The chef cooked a delicious meal.".into(),
            output: String::new(),
        },
    ];

    for e in &samples {
        let text = generate_response(model, e, tk, 40, device);
        println!("\n指令: {}", e.instruction);
        println!("输入: {}", e.input);
        println!("回答: {text:?}");
    }

    println!("\n对比一下数据集里的标准答案 (data[0]):");
    println!("  输入: {}", data[0].input);
    println!("  参考: {}", data[0].output);
    println!("\n→ 只训练了 {} 步，回答还很粗糙。", "少量");
    println!("  书中用 355M 模型跑 5 个 epoch、1100 条数据，才能到比较像样的水平。");
}
