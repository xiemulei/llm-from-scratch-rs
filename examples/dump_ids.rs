//! 诊断工具：打印任意文本的 GPT-2 词元 ID（供 numpy 参考实现使用）。
fn main() {
    let tk = tiktoken_rs::bpe_for_model("gpt2").unwrap();
    let text = std::env::args().nth(1).unwrap_or_else(|| {
        "It is a truth long known to engineers that a model can only learn what its data allows.".to_string()
    });
    println!("{:?}", tk.encode_ordinary(&text));
    for p in ["Every effort moves you", "The capital of France is"] {
        println!("// {p:?} => {:?}", tk.encode_ordinary(p));
    }
}
