---
{
  "id": "p3-transformers",
  "title": "Transformers and Attention",
  "phase": 3,
  "difficulty": "advanced",
  "estimated_hours": 16,
  "objectives": [
    "Understand self-attention mechanism",
    "Explain multi-head attention and positional encoding",
    "Trace the transformer encoder/decoder architecture",
    "Recognize why transformers dominate modern ML"
  ],
  "concepts": [
    "Self-attention: query, key, value",
    "Scaled dot-product attention",
    "Multi-head attention",
    "Positional encoding (sinusoidal, learned)",
    "Encoder-decoder vs encoder-only architectures",
    "Layer normalization and residual connections",
    "BERT, GPT, Vision Transformer (ViT)",
    "Computational complexity O(n²)"
  ],
  "formulas": [
    {"name": "Attention", "latex": "\\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)V", "note": "Weighted sum of values by query-key similarity."},
    {"name": "Multi-head", "latex": "\\text{MH}(Q,K,V) = \\text{Concat}(\\text{head}_1,...,\\text{head}_h)W^O", "note": "Parallel attention in different subspaces."},
    {"name": "Positional encoding", "latex": "PE_{pos,2i} = \\sin(pos/10000^{2i/d}); PE_{pos,2i+1} = \\cos(pos/10000^{2i/d})", "note": "Sinusoidal position information."}
  ],
  "references": [
    "Vaswani et al. 2017 'Attention Is All You Need'",
    "The Illustrated Transformer (Alammar blog)",
    "Goodfellow et al. ch. 10.3",
    "Hugging Face transformers documentation"
  ]
}
---

## Penjelasan

Transformer adalah arsitektur yang menggantikan rekursi dengan *self-attention* — mekanisme yang memungkinkan setiap posisi dalam sequence melihat semua posisi lain secara langsung. Ini menghilangkan masalah sequential processing RNN dan menjadi fondasi GPT, BERT, ViT, dan hampir semua model NLP modern.

**Self-attention.** Setiap token menghasilkan tiga vektor: Query (apa yang dicari), Key (apa yang ditawarkan), Value (informasi aktual). Attention score antara query i dan key j = dot(Q_i, K_j)/√d_k. Softmax -> bobot -> weighted sum values. Intuisinya: setiap token "bertanya" ke token lain dan mengumpulkan informasi relevan.

**Multi-head attention.** Alih-alih satu attention, jalankan h attention paralel di subspace berbeda (Q,K,V diproyeksikan ke d_k = d_model/h dimensi). Setiap head bisa fokus pada pola berbeda: sintaks, semantik, referensi jarak jauh. Output digabung dan diproyeksikan.

**Positional encoding.** Self-attention itu *permutation-invariant* — tidak tahu urutan token. Positional encoding menambahkan informasi posisi: sinusoidal (sin/cos frekuensi berbeda per dimensi) atau learned embeddings. Tanpa ini, "dog bites man" = "man bites dog".

**Arsitektur.** Encoder-only (BERT): bidirectional attention untuk understanding tasks. Decoder-only (GPT): causal masking (hanya lihat masa lalu) untuk generation. Encoder-decoder (original Transformer): untuk translation. Vision Transformer (ViT): patch embedding + transformer encoder untuk gambar.

**Layer norm & residual.** Setiap sub-layer (attention, FFN) diikuti LayerNorm + residual connection — menstabilkan training dalam dan memungkinkan gradient flow. FFN = 2-layer MLP dengan aktivasi (biasanya GELU).

## Konsep Penting

1. **O(n²) complexity** — attention matrix n×n; sequence panjang mahal; mitigasi: sparse attention, linear attention, sliding window.
2. **Causal masking** — decoder hanya attend ke posisi sebelumnya (masking di softmax); essential untuk autoregressive generation.
3. **Pre-norm vs post-norm** — modern transformers pakai pre-LN (LN sebelum sub-layer) untuk stabilitas.
4. **Scaling by √d_k** — mencegah dot products besar -> softmax saturasi -> gradient kecil.
5. **Transfer learning** — pretrained transformer + fine-tune adalah standar NLP/CV modern.

## Formula

- Attention: softmax(QK^T/√d_k)V.
- Multi-head concat: Concat(head_1..h)W^O.
- Positional: PE[pos,2i]=sin(pos/10000^(2i/d)).
- FFN: FFN(x) = max(0, xW1+b1)W2+b2 (atau GELU).

## Contoh Kasus

Machine translation EN->DE: encoder transformer membaca kalimat Inggris, decoder menghasilkan Jerman token per token dengan causal attention ke output sebelumnya + cross-attention ke encoder output. Bandingkan dengan LSTM seq2seq — transformer lebih paralel dan handle long-range dependencies lebih baik.

## Implementasi

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        B, N, _ = Q.shape
        q = self.W_q(Q).view(B, N, self.n_heads, self.d_k).transpose(1,2)
        k = self.W_k(K).view(B, N, self.n_heads, self.d_k).transpose(1,2)
        v = self.W_v(V).view(B, N, self.n_heads, self.d_k).transpose(1,2)
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1,2).reshape(B, N, self.d_model)
        return self.W_o(out)

# sinusoidal positional encoding
def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Self-attention allows each token to:",
    "options": [
      "Directly attend to all other tokens regardless of distance",
      "Process tokens sequentially",
      "Only look at previous tokens",
      "Ignore positional information"
    ],
    "answer": 0,
    "explanation": "Attention computes pairwise interactions between all positions — no sequential bottleneck.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Why is positional encoding needed in transformers?",
    "options": [
      "Self-attention is permutation-invariant; it doesn't know token order",
      "To reduce parameters",
      "To speed up training",
      "To handle variable length inputs"
    ],
    "answer": 0,
    "explanation": "Attention treats tokens as a set; PE injects order information.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "Multi-head attention helps by:",
    "options": [
      "Learning different types of relationships in parallel subspaces",
      "Reducing computation",
      "Increasing model depth",
      "Adding more parameters only"
    ],
    "answer": 0,
    "explanation": "Each head can specialize: syntax, semantics, coreference, etc.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `scaled_dot_product_attention(Q, K, V)` for single head returning attention output (no masking).",
    "starter_code": "import torch\nimport math\n\ndef scaled_dot_product_attention(Q, K, V):\n    d_k = Q.shape[-1]\n    pass\n",
    "function": "scaled_dot_product_attention",
    "tests": [
      {"call": "scaled_dot_product_attention(torch.eye(2),torch.eye(2),torch.eye(2)).tolist()", "expected": "check"}
    ],
    "hints": ["scores = Q@K.T/sqrt(d_k); attn=softmax(scores,dim=-1); return attn@V"],
    "explanation": "Core attention mechanism.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Compare Transformer vs LSTM for a time series forecasting task: which would you choose and why? Consider sequence length, training parallelism, interpretability, and computational cost.",
    "rubric": [
      "Transformer better for long sequences (parallel, no vanishing gradient)",
          "LSTM better for short sequences or when interpretability of hidden state needed",
      "Transformer O(n²) memory; LSTM O(n) but sequential",
      "For most modern applications, Transformer preferred; LSTM for streaming/edge"
        ],
    "points": 2
  }
]
```

## Referensi

- Vaswani et al. (2017) 'Attention Is All You Need'
- 'The Illustrated Transformer' — Jay Alammar blog
- 'The Annotated Transformer' — Harvard NLP
- Hugging Face transformers documentation
- Dosovitskiy et al. (2020) 'An Image is Worth 16x16 Words' (ViT)
