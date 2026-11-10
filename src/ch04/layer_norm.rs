//! 4.2 层归一化（Layer Normalization）。
//!
//! 训练深度网络时，各层激活值的尺度会漂移，导致梯度不稳。归一化把每层输出拉回
//! 均值 0、方差 1 附近，再乘一个可学习的缩放 `γ`、加一个可学习的偏移 `β`。
//!
//! LayerNorm 和 BatchNorm 的区别（书中重点强调）：
//!
//! - **BatchNorm** 在**批量维**上统计：每个特征用整个 batch 的均值/方差。依赖 batch 大小，
//!   推理时还要用移动平均，不适合变长序列。
//! - **LayerNorm** 在**特征维**上统计：每个 token 只用自己的所有特征求均值/方差。
//!   与 batch 无关，序列长度可以任意变化——这正是 NLP 需要的。
//!
//! 书中先手写一遍公式再换成 `nn.LayerNorm`，这里同样保留两份实现，并用测试保证二者一致。

use burn::tensor::Tensor;

/// 4.2 的手写版本：`y = (x - μ) / √(σ² + ε) · γ + β`
///
/// 输入 `[n, d]`（n 个 token，每个 d 维），沿最后一维归一化。
pub fn layer_norm_manual(x: Tensor<2>, gamma: &Tensor<2>, beta: &Tensor<2>, eps: f64) -> Tensor<2> {
    let [n, d] = x.dims();

    // 均值 μ：每个 token 一个标量，扩展回 [n, d] 便于广播相减
    let mean = x.clone().mean_dim(1).expand([n, d]);
    // 方差 σ²：注意这里用的是"总体方差"（除以 d 而不是 d-1），与 PyTorch / Burn 一致
    let var = (x.clone() - mean.clone()).powf_scalar(2.0).mean_dim(1).expand([n, d]);

    let normalized = (x - mean)
        / var
            .add_scalar(eps) // 防止除零
            .sqrt();

    normalized * gamma.clone() + beta.clone()
}

/// 只做归一化、不做仿射变换的版本，用于观察归一化本身的效果。
pub fn normalize_only(x: Tensor<2>, eps: f64) -> Tensor<2> {
    let [n, d] = x.dims();
    let mean = x.clone().mean_dim(1).expand([n, d]);
    let var = (x.clone() - mean.clone()).powf_scalar(2.0).mean_dim(1).expand([n, d]);
    (x - mean) / var.add_scalar(eps).sqrt()
}

/// 计算每一行的均值与方差，用于把书中 4.2 的中间结果打印出来对照。
pub fn row_stats(x: &Tensor<2>, eps: f64) -> (Vec<f32>, Vec<f32>) {
    let [n, d] = x.dims();
    let mean = x.clone().mean_dim(1).to_data().try_to_vec::<f32>().unwrap();
    let var = (x.clone() - x.clone().mean_dim(1).expand([n, d]))
        .powf_scalar(2.0)
        .mean_dim(1)
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    let _ = eps;
    (mean, var)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::nn::LayerNormConfig;
    use burn::tensor::{Device, Distribution};

    #[test]
    fn manual_matches_burn_layer_norm() {
        let device = Device::default();
        let x = Tensor::<2>::random([4, 16], Distribution::Default, &device);

        let mut ln = LayerNormConfig::new(16).init(&device);
        // 让 Burn 的模块用全 1 的 γ、全 0 的 β，才能和手写版本对齐
        ln.gamma = burn::module::Param::from_tensor(Tensor::<1>::ones([16], &device));
        ln.beta = Some(burn::module::Param::from_tensor(Tensor::<1>::zeros([16], &device)));

        let manual = layer_norm_manual(
            x.clone(),
            &Tensor::<2>::ones([1, 16], &device),
            &Tensor::<2>::zeros([1, 16], &device),
            1e-5,
        );
        let expected = ln.forward(x);

        let a = manual.to_data().try_to_vec::<f32>().unwrap();
        let b = expected.to_data().try_to_vec::<f32>().unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-4, "手写 LayerNorm 与 Burn 不一致: {x} vs {y}");
        }
    }

    #[test]
    fn normalized_rows_have_zero_mean_unit_variance() {
        let device = Device::default();
        let x = Tensor::<2>::random([3, 8], Distribution::Default, &device) * 5.0 + 3.0;
        let out = normalize_only(x, 1e-5);
        let (mean, var) = row_stats(&out, 1e-5);
        for (m, v) in mean.iter().zip(var.iter()) {
            assert!(m.abs() < 1e-5, "均值应为 0，实际 {m}");
            assert!((v - 1.0).abs() < 1e-3, "方差应为 1，实际 {v}");
        }
    }

    #[test]
    fn layer_norm_is_batch_independent() {
        let device = Device::default();
        let gamma = Tensor::<2>::ones([1, 8], &device);
        let beta = Tensor::<2>::zeros([1, 8], &device);
        let x = Tensor::<2>::random([2, 8], Distribution::Default, &device);

        let both = layer_norm_manual(x.clone(), &gamma, &beta, 1e-5);
        let only_first = layer_norm_manual(x.slice([0..1]), &gamma, &beta, 1e-5);

        let a = both.slice([0..1]).to_data().try_to_vec::<f32>().unwrap();
        let b = only_first.to_data().try_to_vec::<f32>().unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6, "LayerNorm 不应依赖同批次的其他样本");
        }
    }

    #[test]
    fn learnable_params_change_the_output() {
        let device = Device::default();
        let x = Tensor::<2>::random([2, 4], Distribution::Default, &device);

        // γ 的形状要与输入对齐（Burn 的广播不像 PyTorch 那样自由，这里直接铺满两行）
        let scaled = layer_norm_manual(
            x.clone(),
            &(Tensor::<2>::ones([2, 4], &device) * 2.0),
            &Tensor::<2>::zeros([2, 4], &device),
            1e-5,
        );
        let baseline = normalize_only(x, 1e-5);

        let a = baseline.to_data().try_to_vec::<f32>().unwrap();
        let b = scaled.to_data().try_to_vec::<f32>().unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x * 2.0 - y).abs() < 1e-5);
        }
    }
}
