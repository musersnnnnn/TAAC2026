2. DIN 支路：商品导向的兴趣摘要
item_anchor 从哪里来
你当前选中的代码：
item_anchor = torch.cat(item_parts, dim=1).mean(dim=1)
位置在 [model.py (line 1971)](/home/wyj/project/TAAC/TAAC2026---0.831174/model.py:1971)。
含义是：
商品 NS token 1
商品 NS token 2
（如果有商品连续 token，也包含）
      ↓ mean
item_anchor: [B,64]
它代表“当前要预测的商品”。
DIN 如何读每个序列
DIN 使用当前商品向量作为 query，对每条历史行为 token 打分：
item_anchor q: [B,64]
历史行为 seq_i: [B,L,64]
将 q 扩展到每个历史位置，然后拼接：
[q, seq, q - seq, q * seq]
得到：
[B,L,256]
↓ MLP
[B,L]
↓ mask + softmax
每条历史行为的权重
↓ 使用这些权重和 seq-tokens 加权求和
context_i: [B,64]


四个域各得到一个 context：
context_a, context_b, context_c, context_d: [B,64]
随后用一个 domain gate 决定四个域的重要性：
[item_anchor, context_a, context_b, context_c, context_d]
↓ MLP + softmax
domain 权重
↓
din_context: [B,64]
见 [model.py (line 1335)](/home/wyj/project/TAAC/TAAC2026---0.831174/model.py:1335)。
注意：DIN 在 HyFormer 之前执行，读取的是初始的 Seq token，不是 Transformer 更新后的 Seq token。
