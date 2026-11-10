//! 第 2 章演示：文本 → 词元 → ID → 训练样本 → 嵌入张量。
//!
//! 运行：`cargo run --bin ch02`

use anyhow::Result;
use burn::tensor::Device;
use llm_from_scratch::ch02::{
    bpe::Bpe,
    dataset::{make_batches, GptDataset},
    embeddings::{batch_to_tensors, GptEmbedding},
    gpt2_tokenizer::{Gpt2Tokenizer, GPT2_VOCAB_SIZE},
    simple_tokenizer::{simple_tokenize, SimpleVocab, END_OF_TEXT, UNK},
    SAMPLE_TEXT,
};
use llm_from_scratch::section;

fn main() -> Result<()> {
    demo_simple_tokenizer();
    demo_simple_vocab();
    demo_bpe();
    demo_gpt2()?;
    demo_dataset();
    demo_embeddings();
    Ok(())
}

/// 2.2 用正则（这里是手写规则）切分文本
fn demo_simple_tokenizer() {
    section("2.2 分词：按空白与标点切分");
    let text = "Hello, do you like tea? In the sunlit terraces of the palace.";
    let tokens = simple_tokenize(text);
    println!("原文  : {text}");
    println!("词元数: {}", tokens.len());
    println!("词元  : {tokens:?}");
}

/// 2.3 / 2.4 词表映射与未知词处理
fn demo_simple_vocab() {
    section("2.3-2.4 词表：字符串 ↔ 整数 ID");
    let corpus = format!("{SAMPLE_TEXT} {END_OF_TEXT}");
    let vocab = SimpleVocab::from_corpus(&corpus, &[UNK, END_OF_TEXT]);
    println!("词表大小: {}", vocab.len());

    let text = "The little machine sat on the desk and watched the words.";
    let ids = vocab.encode(text);
    println!("原文: {text}");
    println!("ID  : {ids:?}");
    println!("还原: {}", vocab.decode(&ids));

    // 未登录词 → <|unk|>
    let unknown = "The quantum blockchain sat on the desk.";
    let ids = vocab.encode(unknown);
    println!("\n含未登录词: {unknown}");
    println!("ID  : {ids:?}");
    println!("还原: {}", vocab.decode(&ids));
    println!("→ 词表法的两个死穴：词表爆炸 + 未登录词。BPE 就是来救场的。");
}

/// 2.5 从零训练 BPE
fn demo_bpe() {
    section("2.5 从零实现字节对编码（BPE）");
    let corpus = SAMPLE_TEXT.repeat(8);
    let target_vocab = 400;

    let (bpe, merges) = Bpe::train(&corpus, target_vocab);
    println!("语料字节数      : {}", corpus.len());
    println!("学习的合并规则数: {}", bpe.num_merges());
    println!("最终词表大小    : {}", 256 + bpe.num_merges());

    println!("\n前 10 条合并规则 (左, 右) → 新词元:");
    for (i, (l, r, id)) in bpe.show_merges(10).into_iter().enumerate() {
        println!("  {i:>2}. ({l:?}, {r:?}) → {id}");
    }

    let text = "The little machine sat on the desk.";
    let ids = bpe.encode(text);
    println!("\n编码: {text}");
    println!("字节数 {} → 词元数 {}", text.len(), ids.len());
    println!("ID    : {ids:?}");
    println!("还原  : {}", bpe.decode(&ids));
    let _ = merges;

    // 关键性质：byte-level 表示永不产生未登录词
    let unseen = "完全没见过的中文和 emoji 🦀";
    let ids = bpe.encode(unseen);
    println!("\n未见过的文本: {unseen}");
    println!("仍能无损还原: {}", bpe.decode(&ids) == unseen);
}

/// 2.5（续）GPT-2 官方分词器
fn demo_gpt2() -> Result<()> {
    section("2.5 续：GPT-2 官方 BPE（tiktoken）");
    let tk = Gpt2Tokenizer::new()?;
    println!("词表大小: {GPT2_VOCAB_SIZE}");

    let text = "Hello, do you like tea?";
    let pairs = tk.tokenize_with_ids(text);
    println!("\n{text}");
    println!("词元 → ID:");
    for (tok, id) in &pairs {
        println!("  {tok:<12?} → {id}");
    }

    let full = format!("<|endoftext|> {text}");
    let ids = tk.encode_with_special(&full);
    println!("\n加入文档分隔符后共 {} 个词元", ids.len());
    println!("首词元 ID = {}（<|endoftext|>）", ids[0]);
    println!("解码还原: {:?}", tk.decode(&ids));
    Ok(())
}

/// 2.6 滑动窗口数据集
fn demo_dataset() {
    section("2.6 滑动窗口数据采样");
    let corpus = SAMPLE_TEXT.repeat(8);

    // 先用简易分词器把语料变成 ID 序列（真实训练用 GPT-2 分词器，见 bin/ch05）
    let vocab = SimpleVocab::from_corpus(&corpus, &[UNK, END_OF_TEXT]);
    let tokens: Vec<u32> = vocab.encode(&corpus).iter().map(|&x| x as u32).collect();
    println!("语料词元数: {}", tokens.len());

    println!("\n--- 小例子：max_length=4, stride=1 ---");
    let small: Vec<u32> = (0..12).collect();
    let ds = GptDataset::new(small.clone(), 4, 1);
    println!("token 序列: {small:?}");
    println!("样本数: {}", ds.len());
    for i in 0..ds.len().min(4) {
        let (x, y) = ds.get(i);
        println!("  样本 {i}: input={x:?}  target={y:?}");
    }
    println!("→ target 恒等于 input 右移一位，这就是自监督信号。");

    println!("\n--- 书中实际配置：max_length=256, stride=128, batch=8 ---");
    let ds = GptDataset::new(tokens.clone(), 256, 128);
    println!("样本数: {}（此处语料较短，实际书中约 9 个）", ds.len());

    let max_length = 32;
    let ds = GptDataset::new(tokens, max_length, max_length / 2);
    let batches = make_batches(&ds, 4, true, true);
    println!("使用 max_length={max_length}, batch_size=4 → {} 个 batch", batches.len());
    for (i, b) in batches.iter().take(3).enumerate() {
        println!("  batch {i}: inputs {:?}, targets {:?}", b.batch_size(), b.seq_len());
    }
}

/// 2.7 / 2.8 词元嵌入与位置编码
fn demo_embeddings() {
    section("2.7-2.8 词元嵌入 + 位置编码");
    let device = Device::default();
    println!("设备: {device:?}");

    let vocab_size = 50257;
    let emb_dim = 256;
    let context_length = 64;

    let emb = GptEmbedding::new(vocab_size, emb_dim, context_length, &device);
    println!(
        "词元嵌入矩阵: [{vocab_size}, {emb_dim}]  = {} 个参数",
        vocab_size * emb_dim
    );
    println!(
        "位置嵌入矩阵: [{context_length}, {emb_dim}]  = {} 个参数",
        context_length * emb_dim
    );

    let batch = llm_from_scratch::ch02::dataset::Batch::new(
        vec![vec![15496, 11, 995, 345], vec![40, 1101, 588, 7615]],
        vec![vec![11, 995, 345, 588], vec![1101, 588, 7615, 30]],
    );
    let (inputs, _targets) = batch_to_tensors(&batch, &device);
    println!("\n输入 ID 张量形状: {:?}", inputs.dims());

    let out = emb.forward(inputs);
    println!("嵌入后形状     : {:?}  ([batch, seq_len, emb_dim])", out.dims());

    let first = out
        .clone()
        .slice([0..1, 0..1])
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    println!("\n第一个词元的前 8 维向量:");
    println!("  {:?}", &first[..8.min(first.len())]);

    // 位置编码的作用：同样的词元，位置不同 → 表示不同
    let p = emb.position_weights()
        .slice([0..3])
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    println!(
        "\n位置 0/1/2 嵌入的前 4 维: {:?} / {:?} / {:?}",
        &p[0..4],
        &p[emb_dim..emb_dim + 4],
        &p[2 * emb_dim..2 * emb_dim + 4]
    );
    println!("→ 位置嵌入是可学习参数，随预训练自动调整（GPT-2 的做法）。");
}
