//! 训练步速 benchmark：把「加载权重」和「每步耗时」分开测。
//!
//! ```bash
//! cargo run --release --example bench_step                  # CPU (flex)
//! cargo run --release --features gpu --example bench_step   # GPU (wgpu)
//! ```
//!
//! 测量内容是一次完整的训练步：前向 → 交叉熵 → 反向 → AdamW 更新。
//! 用随机权重（不加载 GPT-2），避免 IO 干扰。

use std::time::Instant;

use burn::optim::{AdamWConfig, GradientsParams};
use burn::tensor::{Int, Tensor, TensorData};
use llm_from_scratch::ch04::gpt::{gpt_config_124m, Gpt};
use llm_from_scratch::ch05::loss::cross_entropy;

fn main() {
    let device = llm_from_scratch::init_device();
    let ad = device.clone().autodiff();

    let batch = 8;
    let seq = 64;
    let steps = 10;
    let warmup = 2;

    println!("构建 GPT-2 124M（随机权重）...");
    let t0 = Instant::now();
    let mut model = Gpt::new(&gpt_config_124m().with_drop_rate(0.0), &ad);
    println!("  构建耗时 {:.2}s", t0.elapsed().as_secs_f64());

    let mut optim = AdamWConfig::new().with_weight_decay(0.1).init();

    // 造一批固定数据，避免随机数生成混进计时
    let x = Tensor::<2, Int>::from_data(TensorData::new(vec![100i64; batch * seq], [batch, seq]), &ad);
    let y = Tensor::<2, Int>::from_data(TensorData::new(vec![200i64; batch * seq], [batch, seq]), &ad);

    println!("预热 {warmup} 步（编译 kernel / 填满缓存）...");
    for _ in 0..warmup {
        let logits = model.forward(x.clone());
        let loss = cross_entropy(logits, y.clone());
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(5e-5, model, grads);
    }

    println!("计时 {steps} 步（batch={batch}, seq={seq}）...");
    let t = Instant::now();
    for _ in 0..steps {
        let logits = model.forward(x.clone());
        let loss = cross_entropy(logits, y.clone());
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(5e-5, model, grads);
    }
    let per_step = t.elapsed().as_secs_f64() / steps as f64;

    let tokens = (batch * seq) as f64;
    println!("\n每步耗时 : {:.1} ms", per_step * 1000.0);
    println!("吞吐     : {:.0} token/s", tokens / per_step);
    println!(
        "等效算力 : ~{:.1} GFLOP/s（按 6·N·T 粗估，N=124M）",
        6.0 * 124e6 * tokens / per_step / 1e9
    );
}
