//! 5.5 加载 GPT-2 官方权重。
//!
//! 这是让模型"会说话"的唯一途径：我们手写的结构和 GPT-2 完全一致，
//! 只要把 OpenAI 训练好的 124M 参数灌进去，立刻就能生成通顺英文。
//!
//! ## 三件麻烦事
//!
//! **1. 权重布局**
//! GPT-2 用的是 `Conv1D`（卷积变体）而不是 `nn.Linear`，权重形状是 `[d_in, d_out]`。
//! 巧的是 Burn 的 `Linear` 也是 `[d_in, d_out]`（PyTorch 的 `nn.Linear` 才是
//! `[d_out, d_in]`），所以**大部分参数可以原样搬运**。唯一的例外是 `lm_head`：
//! HF 存成 `[vocab, d]`，而 Burn 需要 `[d, vocab]`。但我们用了权重共享，
//! 直接用 `wte` 转置，所以这部分根本不需要加载。
//!
//! **2. QKV 三合一**
//! GPT-2 把 query / key / value 的投影合并成一个 `c_attn`：`[768, 2304]`。
//! 必须沿第 1 维切成三份 `[768, 768]`，分别给 `w_query` / `w_key` / `w_value`。
//! bias 同理，`[2304]` 切成三段 `[768]`。
//!
//! **3. 键名映射**
//!
//! | HF (safetensors) | 本项目 |
//! |---|---|
//! | `wte.weight` | `tok_emb.weight` |
//! | `wpe.weight` | `pos_emb.weight` |
//! | `h.{i}.ln_1.{weight,bias}` | `trf_blocks[i].norm1.{gamma,beta}` |
//! | `h.{i}.attn.c_attn.weight` | 切成 `att.{w_query,w_key,w_value}.weight` |
//! | `h.{i}.attn.c_proj.weight` | `att.out_proj.weight` |
//! | `h.{i}.ln_2.*` | `trf_blocks[i].norm2.*` |
//! | `h.{i}.mlp.c_fc.*` | `ff.fc1.*` |
//! | `h.{i}.mlp.c_proj.*` | `ff.fc2.*` |
//! | `ln_f.*` | `final_norm.*` |
//! | `lm_head.weight` | 跳过（与 `wte` 共享） |

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use burn::module::Param;
use burn::nn::LayerNorm;
use burn::tensor::{Device, Tensor, TensorData};
use safetensors::SafeTensors;

use crate::ch04::attention::MultiHeadAttention;
use crate::ch04::feed_forward::FeedForward;
use crate::ch04::gpt::Gpt;
use crate::ch04::transformer_block::TransformerBlock;

/// 下载源，按速度排序。
///
/// - **ModelScope** 实测 ~11 MB/s（国内直连最快）
/// - **hf-mirror** 会 302 跳转到 CDN，速度不稳定
/// - 官方 HuggingFace 在国内经常只有 ~100 KB/s，作为最后兜底
///
/// 文件大小 548,105,171 字节，用来校验是否下完。
pub const GPT2_SOURCES: &[(&str, &str)] = &[
    (
        "ModelScope",
        "https://modelscope.cn/api/v1/models/AI-ModelScope/gpt2/repo?Revision=master&FilePath=model.safetensors",
    ),
    (
        "hf-mirror",
        "https://hf-mirror.com/gpt2/resolve/main/model.safetensors",
    ),
    (
        "HuggingFace",
        "https://huggingface.co/gpt2/resolve/main/model.safetensors",
    ),
];

pub const GPT2_SIZE: u64 = 548_105_171;

/// 已下载的 safetensors 文件。整个文件读进内存（约 550MB）。
pub struct WeightFile {
    data: Vec<u8>,
}

impl WeightFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let data = std::fs::read(path.as_ref())
            .with_context(|| format!("读取失败: {}", path.as_ref().display()))?;
        Ok(Self { data })
    }

    /// 下载到 `dir/gpt2.safetensors`，已存在则直接复用。
    pub fn download(dir: impl AsRef<Path>) -> Result<PathBuf> {
        let path = dir.as_ref().join("gpt2.safetensors");
        if path.exists() {
            println!("已缓存: {}", path.display());
            return Ok(path);
        }
        std::fs::create_dir_all(dir.as_ref())?;

        for (name, url) in GPT2_SOURCES {
            println!("[{name}] 下载 {url}");
            match download_to(url, &path) {
                Ok(()) => {
                    println!("完成: {}", path.display());
                    return Ok(path);
                }
                Err(e) => println!("  失败({e})，尝试下一个源"),
            }
        }
        bail!("所有下载源都失败了，可以手动下载后放到 {}", path.display())
    }

    pub fn names(&self) -> Result<Vec<String>> {
        let st = SafeTensors::deserialize(&self.data)?;
        Ok(st.names().into_iter().map(|s| s.to_string()).collect())
    }

    /// 取一个张量，统一转成 f32（GPT-2 官方权重是 f32，这里兼容 f16 以防万一）。
    pub fn get(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let st = SafeTensors::deserialize(&self.data)?;
        let view = st
            .tensor(name)
            .map_err(|e| anyhow!("找不到张量 {name}: {e}"))?;
        let shape = view.shape().to_vec();
        let raw = view.data();

        let values: Vec<f32> = match view.dtype() {
            safetensors::Dtype::F32 => raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            safetensors::Dtype::F16 => raw
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            other => bail!("暂不支持的数据类型 {other:?}"),
        };
        Ok((values, shape))
    }

    fn tensor2(&self, name: &str, device: &Device) -> Result<Tensor<2>> {
        let (values, shape) = self.get(name)?;
        if shape.len() != 2 {
            bail!("{name} 期望 2 维，实际 {shape:?}");
        }
        Ok(Tensor::<2>::from_data(
            TensorData::new(values, [shape[0], shape[1]]),
            device,
        ))
    }

    fn tensor1(&self, name: &str, device: &Device) -> Result<Tensor<1>> {
        let (values, shape) = self.get(name)?;
        if shape.len() != 1 {
            bail!("{name} 期望 1 维，实际 {shape:?}");
        }
        Ok(Tensor::<1>::from_data(
            TensorData::new(values, [shape[0]]),
            device,
        ))
    }
}

fn tensor2(values: Vec<f32>, shape: [usize; 2], device: &Device) -> Tensor<2> {
    Tensor::<2>::from_data(TensorData::new(values, shape), device)
}

fn tensor1(values: Vec<f32>, n: usize, device: &Device) -> Tensor<1> {
    Tensor::<1>::from_data(TensorData::new(values, [n]), device)
}

/// 带断点续传的下载。
///
/// 国内网络下载 550MB 经常中途断开，所以：
/// 1. 写 `.part` 文件，断了就留着；
/// 2. 下次用 `Range: bytes=N-` 从断点继续；
/// 3. 校验最终大小，不完整就重试。
fn download_to(url: &str, path: &Path) -> Result<()> {
    use std::io::Write;

    let part = path.with_extension("part");
    let mut attempts = 0;

    while attempts < 20 {
        attempts += 1;
        let done = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);

        if done >= GPT2_SIZE {
            break;
        }

        let mut req = ureq::get(url);
        if done > 0 {
            req = req.header("Range", &format!("bytes={done}-"));
        }
        let mut resp = match req.call() {
            Ok(r) => r,
            Err(e) => {
                println!("  连接失败: {e}，重试 {attempts}/20");
                continue;
            }
        };

        // 服务器不认 Range 时会回 200 和完整内容，那就从头写
        let resuming = resp.status().as_u16() == 206;
        let mut file = if resuming {
            std::fs::OpenOptions::new().append(true).open(&part)?
        } else {
            std::fs::File::create(&part)?
        };

        // 流式拷贝：Body::read_to_vec 默认 10MB 上限，必须走 reader
        let mut reader = resp.body_mut().as_reader();
        match std::io::copy(&mut reader, &mut file) {
            Ok(_) => {}
            Err(e) => {
                let _ = file.flush();
                println!("  传输中断({e})，已下载 {} 字节，继续续传", file.metadata()?.len());
            }
        }
        file.flush()?;
    }

    let size = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if size != GPT2_SIZE {
        bail!("下载不完整: {size} / {GPT2_SIZE} 字节，请重跑（会从断点继续）");
    }
    std::fs::rename(&part, path)?;
    Ok(())
}

/// f16 位模式 → f32
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // 非规格化数：规格化后再构造
            let mut e = 127 - 15 + 1;
            let mut m = frac;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(f32_bits)
}

/// 只加载主干（嵌入 + Transformer 块 + final_norm），不碰输出头。
///
/// 第 6 章分类微调会把输出头换成 2 分类，所以不需要加载 lm_head。
pub fn load_gpt2_backbone(weights: &WeightFile, model: &mut Gpt, device: &Device) -> Result<()> {
    let find = |name: &str| -> String {
        // 不同转换工具导出的 key 带有/不带有 `transformer.` 前缀，两种都试
        if weights.get(name).is_ok() {
            name.to_string()
        } else {
            format!("transformer.{name}")
        }
    };

    // ---- 词元嵌入 / 位置嵌入 ----
    let wte = weights.tensor2(&find("wte.weight"), device)?;
    model.tok_emb.weight = Param::from_tensor(wte.clone());
    model.pos_emb.weight = Param::from_tensor(weights.tensor2(&find("wpe.weight"), device)?);

    // ---- 每个 Transformer 块 ----
    let n_layers = model.trf_blocks.len();
    for (i, block) in model.trf_blocks.iter_mut().enumerate() {
        let p = |s: &str| -> String { find(&format!("h.{i}.{s}")) };
        load_block(weights, block, &p, device)
            .with_context(|| format!("加载第 {i} 层失败（共 {n_layers} 层）"))?;
    }

    // ---- 最终 LayerNorm ----
    load_layer_norm(weights, &mut model.final_norm, &find("ln_f.weight"), &find("ln_f.bias"), device)?;

    Ok(())
}

/// 把 GPT-2 官方权重灌进我们的模型（含输出头）。
pub fn load_gpt2(weights: &WeightFile, model: &mut Gpt, device: &Device) -> Result<()> {
    load_gpt2_backbone(weights, model, device)?;

    let find = |name: &str| -> String {
        if weights.get(name).is_ok() {
            name.to_string()
        } else {
            format!("transformer.{name}")
        }
    };

    // ---- 输出头 ----
    if model.is_weight_tied() {
        model.sync_output_head();
    } else if weights.get(&find("lm_head.weight")).is_ok() {
        // HF 的 lm_head 是 [vocab, d]，Burn 的 Linear 要 [d, vocab]
        let lm = weights
            .tensor2(&find("lm_head.weight"), device)?
            .swap_dims(0, 1);
        model.out_head.weight = Param::from_tensor(lm);
    } else {
        model.sync_output_head();
    }

    Ok(())
}

fn load_block(
    w: &WeightFile,
    block: &mut TransformerBlock,
    p: &dyn Fn(&str) -> String,
    device: &Device,
) -> Result<()> {
    // 注意力：c_attn 是 QKV 三合一，要切开
    let (qkv, qkv_shape) = w.get(&p("attn.c_attn.weight"))?;
    let d = qkv_shape[0];
    if qkv_shape != vec![d, 3 * d] {
        bail!("c_attn 形状异常: {qkv_shape:?}");
    }
    let (q, k, v) = split_qkv_data(&qkv, d);
    let (qb, kb, vb) = split_qkv_bias_data(&w.get(&p("attn.c_attn.bias"))?.0);

    let att: &mut MultiHeadAttention = &mut block.att;
    att.w_query.weight = Param::from_tensor(tensor2(q, [d, d], device));
    att.w_key.weight = Param::from_tensor(tensor2(k, [d, d], device));
    att.w_value.weight = Param::from_tensor(tensor2(v, [d, d], device));
    att.w_query.bias = Some(Param::from_tensor(tensor1(qb, d, device)));
    att.w_key.bias = Some(Param::from_tensor(tensor1(kb, d, device)));
    att.w_value.bias = Some(Param::from_tensor(tensor1(vb, d, device)));

    att.out_proj.weight = Param::from_tensor(w.tensor2(&p("attn.c_proj.weight"), device)?);
    att.out_proj.bias = Some(Param::from_tensor(w.tensor1(&p("attn.c_proj.bias"), device)?));

    // 两个 LayerNorm
    load_layer_norm(w, &mut block.norm1, &p("ln_1.weight"), &p("ln_1.bias"), device)?;
    load_layer_norm(w, &mut block.norm2, &p("ln_2.weight"), &p("ln_2.bias"), device)?;

    // 前馈网络
    let ff: &mut FeedForward = &mut block.ff;
    ff.fc1.weight = Param::from_tensor(w.tensor2(&p("mlp.c_fc.weight"), device)?);
    ff.fc1.bias = Some(Param::from_tensor(w.tensor1(&p("mlp.c_fc.bias"), device)?));
    ff.fc2.weight = Param::from_tensor(w.tensor2(&p("mlp.c_proj.weight"), device)?);
    ff.fc2.bias = Some(Param::from_tensor(w.tensor1(&p("mlp.c_proj.bias"), device)?));

    Ok(())
}

fn load_layer_norm(
    w: &WeightFile,
    ln: &mut LayerNorm,
    gamma_name: &str,
    beta_name: &str,
    device: &Device,
) -> Result<()> {
    ln.gamma = Param::from_tensor(w.tensor1(gamma_name, device)?);
    ln.beta = Some(Param::from_tensor(w.tensor1(beta_name, device)?));
    Ok(())
}

/// 从 `[d, 3d]` 的行主序数据里取出第 `i` 块列，得到 `[d, d]`。
///
/// ⚠️ 这里刻意不用 `tensor.slice()`：在 autodiff 设备上 slice 会产生**非叶子张量**，
/// 而 `Param::from_tensor` 只接受叶子张量（否则报
/// "Can't convert a non leaf tensor into a tracked tensor"）。
/// 直接从原始数据重新构造张量，天然就是叶子。
fn take_column_block(values: &[f32], d: usize, i: usize) -> Vec<f32> {
    let stride = 3 * d;
    let mut out = Vec::with_capacity(d * d);
    for row in 0..d {
        let start = row * stride + i * d;
        out.extend_from_slice(&values[start..start + d]);
    }
    out
}

fn split_qkv_data(values: &[f32], d: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    (
        take_column_block(values, d, 0),
        take_column_block(values, d, 1),
        take_column_block(values, d, 2),
    )
}

fn split_qkv_bias_data(values: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let d = values.len() / 3;
    (
        values[0..d].to_vec(),
        values[d..2 * d].to_vec(),
        values[2 * d..3 * d].to_vec(),
    )
}

/// 便捷入口：给定权重文件，构造一个已加载好权重的 GPT-2 small。
pub fn build_gpt2_small(weights: &WeightFile, device: &Device) -> Result<Gpt> {
    let cfg = crate::ch04::gpt::gpt_config_124m()
        .with_tie_weights(true)
        .with_drop_rate(0.0);
    let mut model = Gpt::new(&cfg, device);
    load_gpt2(weights, &mut model, device)?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_conversion_matches_known_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0); // 1.0
        assert_eq!(f16_to_f32(0xc000), -2.0); // -2.0
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!((f16_to_f32(0x3555) - 0.333_251_95).abs() < 1e-6);
    }

    #[test]
    fn split_qkv_partitions_columns() {
        // [d=2, 3d=6] 行主序：每行前三列→q，中间三列→k，后三列→v
        let values = vec![
            0.0f32, 1.0, 10.0, 11.0, 20.0, 21.0, //
            2.0, 3.0, 12.0, 13.0, 22.0, 23.0,
        ];
        let (q, k, v) = split_qkv_data(&values, 2);
        assert_eq!(q, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(k, vec![10.0, 11.0, 12.0, 13.0]);
        assert_eq!(v, vec![20.0, 21.0, 22.0, 23.0]);
    }

    #[test]
    fn split_bias_into_three_parts() {
        let (q, k, v) = split_qkv_bias_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(q, vec![1.0, 2.0]);
        assert_eq!(k, vec![3.0, 4.0]);
        assert_eq!(v, vec![5.0, 6.0]);
    }
}
