//! 6.4 – 6.6：把 GPT-2 改造成垃圾短信分类器。
//!
//! 三个改动：
//!
//! **1. 换输出头**：`[768 → 50257]` 换成 `[768 → 2]`。
//!
//! **2. 取哪个位置的表示**：GPT-2 是因果模型，第 t 个位置的输出只能看到 0..t。
//! 所以最后一个 token 的隐藏状态 = 整句的「摘要」，用它做分类。书里称之为
//! last-token 策略（对比第 7 章指令微调用同样的思路）。
//!
//! **3. 冻结大部分参数**：书里只训练最后一个 Transformer 块、最终 LayerNorm 和新的输出头，
//! 其余 10 层多保持预训练权重不动。理由：
//!
//! - 数据量小（~2000 条），全量微调会严重过拟合；
//! - 训练参数从 124M 降到 ~7M，显存和时间都省一个数量级；
//! - GPT-2 的低层学到的是通用语言特征，本来就不该动。
//!
//! 在 Burn 里冻结就是 `module.set_require_grad(false)`——梯度不下传，
//! 优化器的 `GradientsParams` 里也就没有这些参数，自然不会被更新。

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Int, Tensor};

use crate::ch04::gpt::Gpt;

#[derive(Module, Debug)]
pub struct SpamClassifier {
    gpt: Gpt,
    head: Linear,
}

impl SpamClassifier {
    /// 用已加载好 GPT-2 权重的模型构造分类器。
    pub fn new(gpt: Gpt, device: &Device) -> Self {
        let d = gpt.hidden_dim();
        Self {
            // 预训练的 GPT-2 有 dropout=0.1；微调时书里保持它开着（相当于正则）
            gpt,
            head: LinearConfig::new(d, 2).init(device),
        }
    }

    /// 冻结除最后 `n_blocks` 个 Transformer 块、最终 LayerNorm 和输出头之外的所有参数。
    pub fn freeze_all_but_last(self, n_blocks: usize) -> Self {
        let n_layers = self.gpt.num_blocks();
        let keep_from = n_layers.saturating_sub(n_blocks);

        // 1. 全部冻结
        let mut gpt = self.gpt.set_require_grad(false);

        // 2. 解冻需要训练的部分
        gpt = gpt.unfreeze_from_block(keep_from);
        let head = self.head.set_require_grad(true);

        Self { gpt, head }
    }

    /// `idx: [batch, seq_len]` → `logits: [batch, 2]`
    pub fn forward(&self, idx: Tensor<2, Int>) -> Tensor<2> {
        let [batch, seq_len] = idx.dims();
        assert!(seq_len > 0, "序列长度不能为 0");

        // 1. 跑完 Transformer，拿到 [batch, seq_len, 768]
        let hidden = self.gpt.forward_hidden(idx);

        // 2. 只取最后一个位置的隐藏状态作为整句表示 → [batch, 768]
        let last = hidden
            .slice([0..batch, (seq_len - 1)..seq_len])
            .squeeze_dim::<2>(1);

        // 3. 2 分类输出头 → [batch, 2]
        self.head.forward(last)
    }

    /// 返回当前**需要梯度**的参数量，用来确认冻结是否生效。
    ///
    /// 遍历整个模块树，只统计 `is_require_grad()` 为真的参数。
    pub fn trainable_params(&self) -> usize {
        #[derive(Default)]
        struct Counter {
            n: usize,
        }
        impl burn::module::ModuleVisitor for Counter {
            fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<D>>) {

                if param.is_require_grad() {
                    self.n += param.shape().num_elements();
                }
            }
        }
        let mut c = Counter::default();
        burn::module::Module::visit(self, &mut c);
        c.n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ch04::gpt::GPTConfig;
    use burn::tensor::TensorData;

    fn tiny(device: &Device) -> SpamClassifier {
        let cfg = GPTConfig::new(100, 16, 16, 4, 2).with_drop_rate(0.0);
        SpamClassifier::new(Gpt::new(&cfg, device), device)
    }

    #[test]
    fn output_has_two_classes() {
        let device = Device::default();
        let model = tiny(&device);
        let idx = Tensor::<2, Int>::from_data(TensorData::new(vec![1i64, 2, 3], [1, 3]), &device);
        assert_eq!(model.forward(idx).dims(), [1, 2]);
    }

    #[test]
    fn batch_classification_shape() {
        let device = Device::default();
        let model = tiny(&device);
        let idx = Tensor::<2, Int>::zeros([5, 7], &device);
        assert_eq!(model.forward(idx).dims(), [5, 2]);
    }

    #[test]
    fn freezing_reduces_trainable_params() {
        // 必须在 autodiff 设备上，`is_require_grad()` 才反映真实状态
        let device = Device::default().autodiff();
        let full = tiny(&device);
        let total = full.num_params();

        let frozen = tiny(&device).freeze_all_but_last(1);
        // 冻结后仍然保留全部参数（只是不更新），所以 num_params 不变
        assert_eq!(frozen.num_params(), total);
        // 但需要梯度的参数应该只剩最后一层 + final_norm + head
        assert!(frozen.trainable_params() < total / 2);
        assert!(frozen.trainable_params() > 0);
    }

    #[test]
    fn classifier_uses_last_token_only() {
        // 同一个前缀、不同的尾部，最后一个 token 不同 → 输出应不同
        let device = Device::default();
        let model = tiny(&device);
        let a = Tensor::<2, Int>::from_data(TensorData::new(vec![1i64, 2, 3], [1, 3]), &device);
        let b = Tensor::<2, Int>::from_data(TensorData::new(vec![1i64, 2, 9], [1, 3]), &device);
        let va = model.forward(a).to_data().try_to_vec::<f32>().unwrap();
        let vb = model.forward(b).to_data().try_to_vec::<f32>().unwrap();
        assert_ne!(va, vb);
    }
}

