//! 2.5：从零实现字节对编码（Byte Pair Encoding）。
//!
//! 书里的 BPE 走的是"数据压缩算法"视角：
//!
//! 1. 把每个词拆成字节序列（byte-level，任何 UTF-8 文本都能表示，永不出现未登录词）；
//! 2. 统计所有相邻字节对的词频；
//! 3. 把最高频的一对合并成新符号，记一条合并规则；
//! 4. 重复 2–3，直到词表达到目标大小。
//!
//! 词表大小 = 256（所有字节）+ 合并规则数。GPT-2 是 50000+ 特殊词元，
//! 其中 50000 = 256 字节 + 大量合并规则。
//!
//! 这里实现的是"可训练"版本，用来理解 tiktoken 内部发生了什么；
//! 生产环境直接用 `gpt2_tokenizer.rs` 里的官方词表即可。

use std::collections::HashMap;

/// 一条合并规则：把 `(left, right)` 两个符号合成一个新符号，新符号 ID = 256 + 规则序号。
pub type Merge = (u32, u32);

#[derive(Debug, Clone, Default)]
pub struct Bpe {
    /// 按学习顺序排列的合并规则，索引即优先级（越靠前优先级越高）。
    merges: Vec<Merge>,
}

impl Bpe {
    /// 在语料上训练 BPE。
    ///
    /// - `vocab_size`：目标词表大小，必须 >= 256 + 1。
    /// - 返回值里一般还会给出一条 `merges` 列表，方便外部查看学到了什么。
    pub fn train(corpus: &str, vocab_size: usize) -> (Self, Vec<Merge>) {
        assert!(vocab_size > 256, "词表至少要能容下 256 个字节");

        // ---- 第 0 步：预分词 ----
        // 统计"词 → 出现次数"，避免对同一个词反复计算。
        // 注意每个词都拆成字节，并保留前导空格（GPT-2 的分词约定）。
        let mut word_freqs: HashMap<Vec<u32>, usize> = HashMap::new();
        for chunk in pretokenize(corpus) {
            let ids: Vec<u32> = chunk.bytes().map(|b| b as u32).collect();
            *word_freqs.entry(ids).or_insert(0) += 1;
        }

        // ---- 第 1–4 步：循环合并最高频的相邻对 ----
        let mut merges: Vec<Merge> = Vec::new();
        let mut ranks: HashMap<Merge, u32> = HashMap::new();

        while 256 + merges.len() < vocab_size {
            // 统计所有词里相邻对的加权频次
            let mut pair_counts: HashMap<Merge, usize> = HashMap::new();
            for (word, freq) in &word_freqs {
                for pair in word.windows(2) {
                    *pair_counts.entry((pair[0], pair[1])).or_insert(0) += freq;
                }
            }

            // 取频次最高的一对；平局时取字典序小的，保证结果可复现
            let Some((&best, _)) = pair_counts
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
            else {
                break; // 没有可合并的了（语料太小）
            };

            // 应用到所有词上：一对相邻符号 → 一个 256+rank 的新符号
            let new_id = 256 + merges.len() as u32;
            let mut next_freqs = HashMap::with_capacity(word_freqs.len());
            for (word, freq) in word_freqs {
                let merged = merge_pair(&word, best, new_id);
                *next_freqs.entry(merged).or_insert(0) += freq;
            }
            word_freqs = next_freqs;

            ranks.insert(best, merges.len() as u32);
            merges.push(best);
        }

        (Bpe { merges: merges.clone() }, merges)
    }

    /// 从已有的合并规则恢复一个分词器（等价于加载训练好的词表）。
    pub fn from_merges(merges: Vec<Merge>) -> Self {
        Self { merges }
    }

    pub fn num_merges(&self) -> usize {
        self.merges.len()
    }

    /// 编码：把文本切成字节，然后反复套用学到的合并规则，直到没有规则可套。
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for chunk in pretokenize(text) {
            for id in self.encode_chunk(chunk.as_bytes()) {
                out.push(id);
            }
        }
        out
    }

    fn encode_chunk(&self, bytes: &[u8]) -> Vec<u32> {
        // 用 (pair -> rank) 查表，每次挑优先级最高（rank 最小）的规则合并。
        // 这里为了清晰用简单实现：每轮扫一遍找最小 rank。
        let ranks: HashMap<Merge, u32> = self
            .merges
            .iter()
            .enumerate()
            .map(|(i, &m)| (m, i as u32))
            .collect();

        let mut ids: Vec<u32> = bytes.iter().map(|b| *b as u32).collect();
        while ids.len() > 1 {
            let mut best: Option<(usize, u32)> = None; // (位置, rank)
            for i in 0..ids.len() - 1 {
                if let Some(&rank) = ranks.get(&(ids[i], ids[i + 1])) {
                    if best.map_or(true, |(_, r)| rank < r) {
                        best = Some((i, rank));
                    }
                }
            }
            let Some((i, rank)) = best else { break };
            ids[i] = 256 + rank;
            ids.remove(i + 1);
        }
        ids
    }

    /// 把 ID 还原成人类可读文本（字节级词表的好处：任何 ID 都能还原）。
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            self.expand(id, &mut bytes);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn expand(&self, id: u32, out: &mut Vec<u8>) {
        if id < 256 {
            out.push(id as u8);
        } else {
            let (l, r) = self.merges[(id - 256) as usize];
            self.expand(l, out);
            self.expand(r, out);
        }
    }

    /// 可视化：打印学到的前 N 条合并规则，对应书中 `print(merges)` 的输出。
    pub fn show_merges(&self, n: usize) -> Vec<(String, String, u32)> {
        self.merges
            .iter()
            .take(n)
            .enumerate()
            .map(|(i, &(l, r))| {
                (
                    self.decode(&[l]),
                    self.decode(&[r]),
                    256 + i as u32,
                )
            })
            .collect()
    }
}

/// 预分词：切出"带前导空格的词"或"单个标点"。
///
/// 对应书中的正则 `\s?[\w]+|...`，这是 BPE 的关键细节——
/// 空格跟着后面的词走，所以 " hello" 和 "hello" 是不同的词元。
fn pretokenize(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut pending_space = false;

    for ch in text.chars() {
        if ch.is_whitespace() {
            if !current.is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
            pending_space = true;
        } else if ch.is_alphanumeric() || ch == '\'' {
            if pending_space && !current.is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
            if pending_space {
                current.push(' ');
                pending_space = false;
            }
            current.push(ch);
        } else {
            // 标点单独成块
            if !current.is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
            if pending_space {
                current.push(' ');
                pending_space = false;
            }
            current.push(ch);
            chunks.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn merge_pair(word: &[u32], pair: Merge, new_id: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(word.len());
    let mut i = 0;
    while i < word.len() {
        if i + 1 < word.len() && word[i] == pair.0 && word[i + 1] == pair.1 {
            out.push(new_id);
            i += 2;
        } else {
            out.push(word[i]);
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_merges_and_roundtrips() {
        // 高频重复 "aa aa aa" 会让 "a"+"a" 成为第一条合并规则
        let corpus = "aa aa aa aa bb";
        let (bpe, merges) = Bpe::train(corpus, 260);
        assert!(!merges.is_empty());

        let text = "aa bb cc";
        let ids = bpe.encode(text);
        assert_eq!(bpe.decode(&ids), text);
    }

    #[test]
    fn unseen_text_still_encodes() {
        // byte-level 的核心优势：任何 UTF-8 都能编码，没有 <unk>
        let corpus = "hello world";
        let (bpe, _) = Bpe::train(corpus, 300);
        let text = "完全没见过的中文 🦀";
        assert_eq!(bpe.decode(&bpe.encode(text)), text);
    }

    #[test]
    fn merges_reduce_token_count() {
        let corpus = "the quick brown fox jumps over the lazy dog. ".repeat(50);
        let (bpe, _) = Bpe::train(&corpus, 400);
        let raw = corpus.as_bytes().len();
        let encoded = bpe.encode(&corpus).len();
        assert!(encoded < raw / 2, "BPE 应该显著压缩词元数");
    }
}
