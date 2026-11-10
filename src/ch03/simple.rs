//! 3.3 简化自注意力：没有可学习参数。
//!
//! 目标：让每个词元"看一遍全句"，然后按相关度加权求和得到自己的新表示。
//!
//! 三步：
//! 1. **打分**：两个词元的点积。方向越接近，点积越大 → 越相关。
//! 2. **归一化**：softmax 把分值转成和为 1 的权重。
//! 3. **聚合**：用权重对所有输入做加权求和，得到上下文向量。
//!
//! 这一版的 Q/K/V 都直接取自输入本身，没有投影矩阵，所以只是"演示机制"。

use burn::tensor::Tensor;

/// 手写 softmax，对应书中 `softmax_naive`。
///
/// 注意：真正实现必须像这里一样减去最大值，否则 `exp` 会溢出。
/// （PyTorch 的 `torch.softmax` 内部已经做了，所以书中看不到这一步。）
pub fn softmax_naive(x: Tensor<2>) -> Tensor<2> {
    let [rows, cols] = x.dims();
    let max = x.clone().max_dim(1); // [rows, 1]
    let shifted = x - max.expand([rows, cols]);
    let e = shifted.exp();
    let sum = e.clone().sum_dim(1).expand([rows, cols]);
    e / sum
}

/// 3.3.1：对**单个**查询向量计算注意力权重。
///
/// 返回 `[n]` 的权重向量。
pub fn attention_weights_for_query(inputs: &Tensor<2>, query_index: usize) -> Tensor<2> {
    let d_in = inputs.dims()[1];
    // 取出第 query_index 行，保持二维以便和矩阵乘
    let query = inputs.clone().slice([query_index..query_index + 1, 0..d_in]);
    // [1, d] × [d, n] → [1, n]：与每个词元求点积
    let attn_scores = query.matmul(inputs.clone().swap_dims(0, 1));
    softmax_naive(attn_scores)
}

/// 3.3.2：一次性计算**所有**词元的注意力权重（批量化，用矩阵乘代替循环）。
///
/// ```text
/// scores  = X · Xᵀ          [n, n]   每行是一个查询对所有键的分数
/// weights = softmax(scores) [n, n]   每行和为 1
/// context = weights · X     [n, d]   每行是该词元的上下文向量
/// ```
pub fn simple_self_attention(inputs: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
    let attn_scores = inputs.clone().matmul(inputs.clone().swap_dims(0, 1));
    let attn_weights = softmax_naive(attn_scores);
    let context = attn_weights.clone().matmul(inputs);
    (context, attn_weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ch03::matrix;

    fn book_inputs() -> Tensor<2> {
        matrix(
            vec![
                vec![0.43, 0.15, 0.89],
                vec![0.55, 0.87, 0.66],
                vec![0.57, 0.85, 0.64],
                vec![0.22, 0.58, 0.33],
                vec![0.77, 0.25, 0.10],
                vec![0.05, 0.80, 0.55],
            ],
            &burn::tensor::Device::default(),
        )
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let device = burn::tensor::Device::default();
        let x = matrix(vec![vec![1.0, 2.0, 3.0], vec![0.0, 0.0, 0.0]], &device);
        let s = softmax_naive(x);
        let sums = s.sum_dim(1).to_data().try_to_vec::<f32>().unwrap();
        for sum in sums {
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn softmax_is_numerically_stable() {
        let device = burn::tensor::Device::default();
        // 不作 max 减法的实现会在这里溢出成 NaN
        let x = matrix(vec![vec![1000.0, 1001.0]], &device);
        let s = softmax_naive(x).to_data().try_to_vec::<f32>().unwrap();
        assert!(s.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn context_shape_matches_input() {
        let (context, weights) = simple_self_attention(book_inputs());
        assert_eq!(context.dims(), [6, 3]);
        assert_eq!(weights.dims(), [6, 6]);
    }

    #[test]
    fn attention_rows_are_distributions() {
        let (_, weights) = simple_self_attention(book_inputs());
        let sums = weights.sum_dim(1).to_data().try_to_vec::<f32>().unwrap();
        for sum in sums {
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }
}
