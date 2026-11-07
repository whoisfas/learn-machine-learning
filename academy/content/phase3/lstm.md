---
{
  "id": "p3-lstm",
  "title": "LSTM and GRU",
  "phase": 3,
  "difficulty": "advanced",
  "estimated_hours": 14,
  "objectives": [
    "Understand gating mechanisms in LSTM",
    "Explain how gates solve vanishing gradient",
    "Compare LSTM vs GRU architectures",
    "Implement LSTM in PyTorch for sequence tasks"
  ],
  "concepts": [
    "LSTM cell state (constant error carousel)",
    "Forget gate, input gate, output gate",
    "Cell state vs hidden state",
    "GRU: simplified 2-gate variant",
    "Additive vs multiplicative interactions",
    "Peephole connections",
    "Sequence-to-sequence with LSTM encoder-decoder",
    "When to use LSTM vs Transformer"
  ],
  "formulas": [
    {"name": "Forget gate", "latex": "f_t = \\sigma(W_f [h_{t-1}, x_t] + b_f)", "note": "Decides what to forget from cell state."},
    {"name": "Input gate", "latex": "i_t = \\sigma(W_i [h_{t-1}, x_t] + b_i); \\tilde{C}_t = \\tanh(W_C [h_{t-1}, x_t] + b_C)", "note": "What new info to store."},
    {"name": "Cell update", "latex": "C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t", "note": "Additive update — gradient highway."},
    {"name": "Output gate", "latex": "o_t = \\sigma(W_o [h_{t-1}, x_t] + b_o); h_t = o_t \\odot \\tanh(C_t)", "note": "What to output."}
  ],
  "references": [
    "Hochreiter & Schmidhuber 1997 'Long Short-Term Memory'",
    "Cho et al. 2014 'Learning Phrase Representations with GRU'",
    "Goodfellow et al. ch. 10.2",
    "PyTorch nn.LSTM, nn.GRU documentation"
  ]
}
---

## Penjelasan

LSTM (Long Short-Term Memory) adalah RNN variant yang memecahkan masalah vanishing gradient melalui *gating mechanism* dan *cell state* — jalur informasi yang bisa mengalir tanpa perubahan selama ratusan timestep.

**Intuisi.** RNN biasa meng-update hidden state h_t = tanh(...) setiap timestep — informasi lama "terdilusi" oleh transformasi baru. LSTM memisahkan *cell state* C_t (memori jangka panjang, jalur konstan) dari *hidden state* h_t (output ke layer berikutnya). Cell state di-update secara *additive*: C_t = f_t·C_{t-1} + i_t·C̃_t — penambahan, bukan perkalian, sehingga gradient tidak menyusut.

**Tiga gerbang (gates).**
- *Forget gate* f_t = σ(W_f·[h_{t-1},x_t]+b_f): menentukan bagian cell state mana yang dilupakan (0=hapus, 1=simpan).
- *Input gate* i_t = σ(...): menentukan info baru mana yang ditulis ke cell state; C̃_t = tanh(...) adalah kandidat nilai baru.
- *Output gate* o_t = σ(...): menentukan bagian cell state mana yang dikeluarkan sebagai h_t = o_t·tanh(C_t).

**Mengapa bekerja.** Cell state C_t diperbarui secara additive: gradient ∂C_t/∂C_{t-1} = f_t (antara 0 dan 1). Jika forget gate belajar mempertahankan f_t≈1, gradient mengalir tanpa penyusutan — inilah "constant error carousel". LSTM bisa mengingat informasi selama ratusan timestep.

**GRU (Gated Recurrent Unit).** Varian lebih sederhana dengan 2 gate: *reset gate* r_t (apa yang dilupakan dari h_{t-1}) dan *update gate* z_t (seberapa banyak update). GRU menggabungkan cell state dan hidden state menjadi satu. Lebih sedikit parameter, sering performa mirip LSTM; lebih cepat untuk data kecil.

## Konsep Penting

1. **Cell state vs hidden state** — C_t adalah memori internal jangka panjang; h_t adalah output yang terlihat layer lain.
2. **Additive update** — C_t = f·C_old + i·C_new; additive (bukan multiplicative) menjaga gradient.
3. **Forget gate initialization** — bias forget gate ke 1 (bukan 0) membantu di awal training untuk mengingat.
4. **GRU vs LSTM** — GRU lebih sederhana/cepat; LSTM lebih ekspresif untuk task kompleks; keduanya jauh lebih baik dari vanilla RNN.
5. **BPTT masih ada** — LSTM/GRU mengurangi tapi tidak menghilangkan masalah gradient; sequence sangat panjang masih sulit.

## Formula

- C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t.
- h_t = o_t ⊙ tanh(C_t).
- GRU: z_t = σ(W_z[h,x]); r_t = σ(W_r[h,x]); h_t = (1-z_t)⊙h_{t-1} + z_t⊙tanh(W[h_{t-1}⊙r_t, x]).

## Contoh Kasus

Machine translation EN->FR: encoder LSTM membaca kalimat Inggris, menghasilkan context vector; decoder LSTM menghasilkan kata Perancis satu per satu. LSTM bisa "mengingat" subjek kalimat dari awal hingga akhir meski kalimat panjang.

## Implementasi

```python
import torch
import torch.nn as nn

# LSTM for sentiment analysis
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                            num_layers=2, dropout=0.3, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional

    def forward(self, x, lengths=None):
        emb = self.embed(x)
        out, (h_n, c_n) = self.lstm(emb)
        # concatenate last forward and backward hidden states
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(h)

# character-level language model with LSTM
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h=None):
        emb = self.embed(x)
        out, h = self.lstm(emb, h)
        return self.fc(out), h
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "LSTM's cell state C_t is updated via:",
    "options": [
      "Additive combination: f_t*C_{t-1} + i_t*C̃_t",
      "Multiplicative only",
      "Reset every timestep",
      "Same as hidden state"
    ],
    "answer": 0,
    "explanation": "Additive update preserves gradient flow; forget gate controls how much old info persists.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "The forget gate in LSTM determines:",
    "options": [
      "What information to discard from cell state",
      "What new information to add",
      "What to output",
      "The learning rate"
    ],
    "answer": 0,
    "explanation": "f_t ∈ [0,1] controls retention of C_{t-1}.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "GRU differs from LSTM by:",
    "options": [
      "Using 2 gates (reset, update) instead of 3, and merging cell/hidden state",
      "Using no gates",
      "Being bidirectional only",
      "Having more parameters"
    ],
    "answer": 0,
    "explanation": "GRU is simpler: reset+update gates, single state.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `lstm_gates(x_t, h_prev, C_prev, params)` computing forget/input/output gates and new cell/hidden states for one timestep.",
    "starter_code": "import numpy as np\n\ndef lstm_gates(x_t, h_prev, C_prev, params):\n    # params: dict with W_f,b_f,W_i,b_i,W_C,b_C,W_o,b_o (all concatenated [h,x])\n    pass\n",
    "function": "lstm_gates",
    "tests": [
      {"call": "lstm_gates([1],[0],[0],{'W_f':[[0]],'b_f':[0],'W_i':[[0]],'b_i':[0],'W_C':[[0]],'b_C':[0],'W_o':[[0]],'b_o':[0]})", "expected": "check"}
    ],
    "hints": ["Concatenate [h_prev, x_t]; compute f=i=o=sigmoid(W*[h,x]+b); C_tilde=tanh(W_C*[h,x]+b_C); C = f*C_prev + i*C_tilde; h = o*tanh(C)", "return (h_new, C_new)"],
    "explanation": "Core LSTM cell computation.",
    "points": 3
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "For a document classification task with documents up to 10,000 words, compare LSTM vs Transformer (BERT-style) approaches: accuracy potential, training cost, inference speed, and when you'd choose each.",
    "rubric": [
      "Transformer generally higher accuracy on long docs; parallel training",
          "LSTM cheaper inference for streaming/low-latency",
      "Transformer needs positional encoding + attention; LSTM processes sequentially",
      "For most NLP tasks today, Transformer preferred; LSTM for edge/streaming"
        ],
    "points": 2
  }
]
```

## Referensi

- Hochreiter & Schmidhuber (1997) 'Long Short-Term Memory'
- Cho et al. (2014) 'Learning Phrase Representations with GRU'
- Goodfellow et al. ch. 10.2
- PyTorch nn.LSTM, nn.GRU documentation
- Colah's blog 'Understanding LSTM Networks' (excellent visual)
