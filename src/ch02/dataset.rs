//! 2.6：滑动窗口数据采样。
//!
//! 语言模型的训练样本是这样构造的：
//!
//! ```text
//! 文本: [t0, t1, t2, t3, t4, t5, t6, ...]
//! max_length = 4, stride = 1
//!
//! 样本 0:  input = [t0 t1 t2 t3]   target = [t1 t2 t3 t4]
//! 样本 1:  input = [t1 t2 t3 t4]   target = [t2 t3 t4 t5]
//! ...
//! ```
//!
//! 关键点：**target 就是 input 整体右移一位**。这就是"自监督"的全部秘密——
//! 不需要任何人工标注，文本自己就是标签。
//!
//! `stride` 控制样本重叠程度：stride = 1 几乎全重叠（过拟合风险高但样本多），
//! stride = max_length 完全不重叠（书中实际训练用 128 vs 256）。

/// GPT 风格的滑动窗口数据集。
///
/// 对应书中 `GPTDatasetV1`，但它直接持有一整份 token id，避免每次取样本都重新分词。
pub struct GptDataset {
    tokens: Vec<u32>,
    max_length: usize,
    stride: usize,
}

impl GptDataset {
    pub fn new(tokens: Vec<u32>, max_length: usize, stride: usize) -> Self {
        assert!(max_length >= 2, "上下文长度至少为 2");
        assert!(stride >= 1, "stride 至少为 1");
        Self {
            tokens,
            max_length,
            stride,
        }
    }

    /// 样本数。窗口需要 `max_length + 1` 个 token（input 占 max_length，target 右移一位）。
    pub fn len(&self) -> usize {
        if self.tokens.len() < self.max_length + 1 {
            return 0;
        }
        // 等价于书中 while 循环里 start < len - max_length 的迭代次数
        (self.tokens.len() - self.max_length - 1) / self.stride + 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// 取第 `index` 个样本：`(input_ids, target_ids)`，长度都是 `max_length`。
    pub fn get(&self, index: usize) -> (Vec<u32>, Vec<u32>) {
        let start = index * self.stride;
        let end = start + self.max_length;
        let input = self.tokens[start..end].to_vec();
        let target = self.tokens[start + 1..end + 1].to_vec();
        (input, target)
    }

    pub fn num_tokens(&self) -> usize {
        self.tokens.len()
    }
}

/// 把若干样本拼成一个 batch，得到两个 `[batch_size, seq_len]` 的矩阵数据。
///
/// 书中这一步由 `DataLoader(batch_size=8, shuffle=True, drop_last=True)` 完成。
/// 我们把它显式写出来，方便后面直接喂给 Burn 的 `Tensor`。
pub struct Batch {
    pub inputs: Vec<Vec<u32>>,
    pub targets: Vec<Vec<u32>>,
}

impl Batch {
    pub fn new(inputs: Vec<Vec<u32>>, targets: Vec<Vec<u32>>) -> Self {
        Self { inputs, targets }
    }

    pub fn batch_size(&self) -> usize {
        self.inputs.len()
    }

    pub fn seq_len(&self) -> usize {
        self.inputs.first().map_or(0, |v| v.len())
    }

    /// 把 `Vec<Vec<u32>>` 拍平成 `Vec<u32>` + 形状，喂给 `TensorData::new`。
    pub fn to_flat(&self) -> (Vec<i64>, Vec<i64>, [usize; 2]) {
        let b = self.batch_size();
        let t = self.seq_len();
        let inputs = self.inputs.iter().flatten().map(|&x| x as i64).collect();
        let targets = self.targets.iter().flatten().map(|&x| x as i64).collect();
        (inputs, targets, [b, t])
    }
}

/// 模拟 `DataLoader(shuffle=True, drop_last=True)`：把所有样本切成 batch。
///
/// 用一个确定性的简易洗牌（线性同余）代替随机数，方便复现实验。
pub fn make_batches(ds: &GptDataset, batch_size: usize, shuffle: bool, drop_last: bool) -> Vec<Batch> {
    let n = ds.len();
    let mut order: Vec<usize> = (0..n).collect();
    if shuffle {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        for i in (1..n).rev() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (state >> 33) as usize % (i + 1);
            order.swap(i, j);
        }
    }

    let full = n / batch_size;
    let take = if drop_last { full * batch_size } else { n };

    order[..take]
        .chunks(batch_size)
        .map(|chunk| {
            let mut inputs = Vec::with_capacity(chunk.len());
            let mut targets = Vec::with_capacity(chunk.len());
            for &i in chunk {
                let (x, y) = ds.get(i);
                inputs.push(x);
                targets.push(y);
            }
            Batch::new(inputs, targets)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(n: usize) -> Vec<u32> {
        (0..n as u32).collect()
    }

    #[test]
    fn target_is_input_shifted_by_one() {
        let ds = GptDataset::new(tokens(10), 4, 1);
        let (x, y) = ds.get(2);
        assert_eq!(x, vec![2, 3, 4, 5]);
        assert_eq!(y, vec![3, 4, 5, 6]);
    }

    #[test]
    fn length_matches_the_book() {
        // 书中例子：stride 1、max_length 4、9 个 token → 5 个样本
        let ds = GptDataset::new(tokens(9), 4, 1);
        assert_eq!(ds.len(), 5);
    }

    #[test]
    fn stride_controls_overlap() {
        let ds = GptDataset::new(tokens(20), 4, 4);
        let (x0, _) = ds.get(0);
        let (x1, _) = ds.get(1);
        assert_eq!(x0, vec![0, 1, 2, 3]);
        assert_eq!(x1, vec![4, 5, 6, 7]);
    }

    #[test]
    fn too_short_corpus_yields_no_samples() {
        let ds = GptDataset::new(tokens(4), 4, 1);
        assert!(ds.is_empty());
    }

    #[test]
    fn batching_shapes_are_correct() {
        let ds = GptDataset::new(tokens(128), 16, 16);
        let batches = make_batches(&ds, 4, false, true);
        assert!(!batches.is_empty());
        for b in &batches {
            assert_eq!(b.batch_size(), 4);
            assert_eq!(b.seq_len(), 16);
        }
    }
}
