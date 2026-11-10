"""独立的 GPT-2 参考实现（numpy），用来交叉验证 Rust/Burn 版本。

用法：
    python tools/verify_numpy.py data/gpt2.safetensors

它只依赖 numpy，直接读 safetensors 文件，手写一遍 forward，
打印几个提示词的 top-5 预测和最终 logits 的统计量。
如果 Rust 版本和这里的输出一致，说明实现正确。
"""

import json
import struct
import sys

import numpy as np

CONFIG = {
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "n_ctx": 1024,
    "vocab": 50257,
    "eps": 1e-5,
}


def load_safetensors(path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        buf = f.read()
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        dtype = info["dtype"]
        arr = np.frombuffer(buf[start:end], dtype=np.float32 if dtype == "F32" else np.float16)
        tensors[name] = arr.reshape(info["shape"]).astype(np.float32)
    return tensors


def layer_norm(x, g, b, eps=CONFIG["eps"]):
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)  # 总体方差，与 PyTorch 一致
    return (x - mean) / np.sqrt(var + eps) * g + b


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def forward(tokens, w):
    T = len(tokens)
    h = w["n_head"]
    d = w["n_embd"]
    hd = d // h

    x = w["wte.weight"][tokens] + w["wpe.weight"][:T]

    mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)

    for i in range(w["n_layer"]):
        p = f"h.{i}."
        # ---- 注意力 ----
        a = layer_norm(x, w[p + "ln_1.weight"], w[p + "ln_1.bias"])
        qkv = a @ w[p + "attn.c_attn.weight"] + w[p + "attn.c_attn.bias"]
        q, k, v = np.split(qkv, 3, axis=-1)

        def split_heads(t):
            return t.reshape(T, h, hd).transpose(1, 0, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scores = q @ k.transpose(0, 2, 1) / np.sqrt(hd) + mask
        attn = softmax(scores) @ v  # [h, T, hd]
        attn = attn.transpose(1, 0, 2).reshape(T, d)
        x = x + attn @ w[p + "attn.c_proj.weight"] + w[p + "attn.c_proj.bias"]

        # ---- 前馈 ----
        m = layer_norm(x, w[p + "ln_2.weight"], w[p + "ln_2.bias"])
        m = gelu(m @ w[p + "mlp.c_fc.weight"] + w[p + "mlp.c_fc.bias"])
        x = x + m @ w[p + "mlp.c_proj.weight"] + w[p + "mlp.c_proj.bias"]

    x = layer_norm(x, w["ln_f.weight"], w["ln_f.bias"])
    # GPT-2 权重共享：输出头就是词元嵌入
    logits = x @ w["wte.weight"].T
    return logits


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/gpt2.safetensors"
    raw = load_safetensors(path)
    w = {k: v for k, v in raw.items()}
    w["n_layer"], w["n_head"], w["n_embd"] = 12, 12, 768

    prompts = {
        "Every effort moves you": [6109, 3626, 6100, 345],
        "The sky is": [464, 6766, 318],
        "The capital of France is": [464, 3139, 286, 4881, 318],
        "1, 2, 3, 4, 5,": [16, 11, 362, 11, 513, 11, 604, 11, 642, 11],
    }

    for text, ids in prompts.items():
        logits = forward(ids, w)[-1]
        top = np.argsort(-logits)[:5]
        print(f"\n{text!r}")
        print(f"  logits: min={logits.min():.3f} max={logits.max():.3f} mean={logits.mean():.3f}")
        print("  top-5: " + " ".join(f"{i}({logits[i]:.1f})" for i in top))


if __name__ == "__main__":
    main()
