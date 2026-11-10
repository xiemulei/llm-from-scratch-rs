//! 第 3 章演示：从简化自注意力到多头因果注意力。
//!
//! 运行：`cargo run --bin ch03`

use burn::module::Module;
use burn::tensor::{Device, Distribution, Tensor};
use llm_from_scratch::ch03::{
    causal::CausalSelfAttention,
    heatmap,
    matrix,
    matrix_string,
    multi_head::{MultiHeadAttention, MultiHeadAttentionWrapper},
    self_attention::SelfAttention,
    simple::{attention_weights_for_query, simple_self_attention},
};
use llm_from_scratch::section;

/// 书中 3.3 的输入：6 个词元（Your / journey / starts / with / one / step），每个 3 维。
fn book_inputs(device: &Device) -> Tensor<2> {
    matrix(
        vec![
            vec![0.43, 0.15, 0.89],
            vec![0.55, 0.87, 0.66],
            vec![0.57, 0.85, 0.64],
            vec![0.22, 0.58, 0.33],
            vec![0.77, 0.25, 0.10],
            vec![0.05, 0.80, 0.55],
        ],
        device,
    )
}

fn main() {
    let device = Device::default();
    println!("设备: {device:?}");

    demo_3_3(&device);
    demo_3_4(&device);
    demo_3_5(&device);
    demo_3_6(&device);
}

/// 3.3 无参数的自注意力
fn demo_3_3(device: &Device) {
    section("3.3 简化自注意力（无可学习参数）");
    let inputs = book_inputs(device);
    println!("输入 [6, 3]:\n{}", matrix_string(&inputs));

    // 3.3.1 单个查询
    let w2 = attention_weights_for_query(&inputs, 1);
    println!("词元 1 对全部词元的注意力权重:");
    println!("{}", matrix_string(&w2));
    println!(
        "  权重和 = {:.4}",
        w2.sum_dim(1).to_data().try_to_vec::<f32>().unwrap()[0]
    );

    // 3.3.2 全部查询 → 上下文向量
    let (context, weights) = simple_self_attention(inputs);
    println!("\n注意力权重矩阵 [6, 6]:");
    println!("{}", matrix_string(&weights));
    println!("上下文向量 [6, 3]（每行 = 全句按权重的加权和）:");
    println!("{}", matrix_string(&context));
    println!("→ 每一行都是原始 6 个向量的凸组合，权重和为 1。");
}

/// 3.4 带可学习投影的自注意力
fn demo_3_4(device: &Device) {
    section("3.4 带可学习权重的自注意力（Wq / Wk / Wv）");
    let inputs = book_inputs(device);

    for (name, attn) in [
        (
            "v1：手工均匀初始化 U(0,1)",
            SelfAttention::new_manual_init(3, 2, device),
        ),
        (
            "v2：Linear 默认初始化（Kaiming 均匀）",
            SelfAttention::new(3, 2, device),
        ),
    ] {
        let (context, weights) = attn.forward(inputs.clone());
        println!("\n{name}");
        println!("参数数量: {}", Module::num_params(&attn));
        println!("上下文 [6, 2]:\n{}", matrix_string(&context));
        println!("注意力权重 [6, 6]:");
        println!("{}", matrix_string(&weights));

        // 注意力分布的"尖锐程度"：越接近 1/6 越均匀
        let w = weights.to_data().try_to_vec::<f32>().unwrap();
        let max_w = w.iter().cloned().fold(f32::MIN, f32::max);
        println!("最大权重 {max_w:.4}（均匀分布应为 {:.4}）", 1.0 / 6.0);
    }
    println!("\n→ 初始化尺度不当会让所有位置的注意力趋同，梯度难以区分不同词元。");
    println!("  这就是书中从 v1 换成 nn.Linear 的原因。");
}

/// 3.5 因果掩码
fn demo_3_5(device: &Device) {
    section("3.5 因果注意力：不能偷看未来");
    let context_length = 6;
    let attn = CausalSelfAttention::new(3, 3, context_length, 0.0, device);

    // 把 [6, 3] 变成 [1, 6, 3]
    let x = book_inputs(device).unsqueeze_dim::<3>(0);
    let (context, weights) = attn.forward(x);

    println!("注意力权重 [1, 6, 6]：");
    println!("{}", matrix_string(&weights.clone().squeeze_dim::<2>(0)));
    println!("热力图（越亮越关注，右上角全黑 = 未来被屏蔽）:");
    println!("{}", heatmap(&weights.clone().squeeze_dim::<2>(0)));

    let w = weights.to_data().try_to_vec::<f32>().unwrap();
    let mut masked = 0;
    for i in 0..6 {
        for j in 0..6 {
            if j > i && w[i * 6 + j].abs() < 1e-6 {
                masked += 1;
            }
        }
    }
    println!("被屏蔽的位置: {masked} / 15（上三角数量 = 6×5/2）");
    println!("上下文形状: {:?}", context.dims());
    println!("→ 第 0 个词元只能看自己，权重恒为 1；第 5 个能看全部 6 个。");
}

/// 3.6 多头注意力
fn demo_3_6(device: &Device) {
    section("3.6 多头注意力");

    let (d_in, d_out, ctx_len, n_heads) = (768, 768, 1024, 12);
    let batch = 2;
    let seq_len = 6;

    let x = Tensor::<3>::random([batch, seq_len, d_in], Distribution::Default, device);
    println!("输入: [batch={batch}, seq_len={seq_len}, d_in={d_in}]");

    // --- 3.6.1 朴素堆叠 ---
    let naive = MultiHeadAttentionWrapper::new(d_in, d_out, ctx_len, 0.1, n_heads, device);
    let out = naive.forward(x.clone());
    println!("\n3.6.1 朴素堆叠（n_heads 个独立注意力 + 拼接）");
    println!("  输出形状: {:?}", out.dims());
    println!("  参数量  : {}", Module::num_params(&naive));

    // --- 3.6.2 权重切分 ---
    let mha = MultiHeadAttention::new(d_in, d_out, ctx_len, 0.1, n_heads, device);
    let (out, weights) = mha.forward(x);
    println!("\n3.6.2 权重切分（GPT-2 的做法）");
    println!("  输出形状: {:?}", out.dims());
    println!("  头数 / 每头维度: {} / {}", mha.n_heads(), mha.head_dim());
    println!("  参数量  : {}", Module::num_params(&mha));
    println!(
        "  → 参数量是朴素版的 1/{n_heads}，因为每个头只投影到 head_dim 而不是 d_out"
    );

    println!("\n  各头的注意力权重 [batch=0, 全部头, 6, 6]:");
    for h in 0..3 {
        let head_w = weights
            .clone()
            .slice([0..1, h..h + 1])
            .squeeze_dim::<3>(1)
            .squeeze_dim::<2>(0);
        println!("\n  头 {h}:");
        println!("{}", heatmap(&head_w));
    }
    println!("→ 不同头学到了不同的关注模式（这里的权重是随机初始化的，训练后才会有意义）。");

    // 说明 d_out % n_heads 的约束
    println!("\n  约束：d_out 必须能被 n_heads 整除，否则无法均分。");
    println!("  GPT-2 small: 768 / 12 = 64 维/头");
}
