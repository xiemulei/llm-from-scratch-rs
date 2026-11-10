//! 第 3 章：注意力机制。
//!
//! 全书最核心的一章。注意力解决的是 RNN/CNN 的两个痛点：
//!
//! - RNN 必须顺序处理，长距离依赖会衰减，且无法并行；
//! - CNN 的"感受野"需要堆很多层才能覆盖整句。
//!
//! 注意力让任意两个位置直接交互（路径长度为 1），并且整个过程是矩阵乘法，可完全并行。
//!
//! 本章按书中的四个渐进版本实现：
//!
//! | 小节 | 版本 | 是否有可学习参数 | 是否因果 |
//! |---|---|---|---|
//! | 3.3 | 简化自注意力 | ❌ | ❌ |
//! | 3.4 | 带 Wq/Wk/Wv 的自注意力 | ✅ | ❌ |
//! | 3.5 | 因果（掩码）自注意力 | ✅ | ✅ |
//! | 3.6 | 多头注意力 | ✅ | ✅ |

pub mod causal;
pub mod multi_head;
pub mod self_attention;
pub mod simple;

use burn::tensor::{Int, Tensor, TensorData};

/// 把 `[rows, cols]` 的二维张量格式化成带两位小数的文本表格，方便和书中的打印对照。
pub fn matrix_string(t: &Tensor<2>) -> String {
    let [rows, cols] = t.dims();
    let data = t.clone().to_data().try_to_vec::<f32>().unwrap();
    let mut out = String::new();
    for r in 0..rows {
        out.push_str("  [");
        for c in 0..cols {
            if c > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!("{:7.3}", data[r * cols + c]));
        }
        out.push_str(" ]\n");
    }
    out
}

/// 把注意力权重画成 ASCII 热力图。
///
/// `chars` 从暗到亮，对应权重从小到大。因果注意力的效果会非常直观：
/// 右上角（未来的位置）永远是空白。
pub fn heatmap(t: &Tensor<2>) -> String {
    const CHARS: &[u8] = b" .:-=+*#%@";
    let [rows, cols] = t.dims();
    let data = t.clone().to_data().try_to_vec::<f32>().unwrap();
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max).max(1e-9);

    let mut out = String::new();
    for r in 0..rows {
        out.push_str("  ");
        for c in 0..cols {
            let v = data[r * cols + c] / max;
            let idx = (v.clamp(0.0, 1.0) * (CHARS.len() - 1) as f32).round() as usize;
            out.push(CHARS[idx] as char);
            out.push(' ');
        }
        out.push('\n');
    }
    out
}

/// 从 `Vec<Vec<f32>>` 构造二维张量的快捷函数。
pub fn matrix(rows: Vec<Vec<f32>>, device: &burn::tensor::Device) -> Tensor<2> {
    let n = rows.len();
    let m = rows.first().map_or(0, |r| r.len());
    let flat: Vec<f32> = rows.into_iter().flatten().collect();
    Tensor::<2>::from_data(TensorData::new(flat, [n, m]), device)
}

/// 从二维整数构造张量，用于把词元 ID 喂给注意力做演示。
pub fn int_matrix(rows: Vec<Vec<i64>>, device: &burn::tensor::Device) -> Tensor<2, Int> {
    let n = rows.len();
    let m = rows.first().map_or(0, |r| r.len());
    let flat: Vec<i64> = rows.into_iter().flatten().collect();
    Tensor::<2, Int>::from_data(TensorData::new(flat, [n, m]), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_roundtrip_shape() {
        let device = burn::tensor::Device::default();
        let m = matrix(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device);
        assert_eq!(m.dims(), [2, 2]);
        assert!(heatmap(&m).lines().count() == 2);
    }
}
