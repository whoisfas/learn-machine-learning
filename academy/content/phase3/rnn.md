---
{
  "id": "p3-rnn",
  "title": "Recurrent Neural Networks",
  "phase": 3,
  "difficulty": "advanced",
  "estimated_hours": 14,
  "objectives": [
    "Understand sequential data processing with RNNs",
    "Explain hidden state and temporal dependencies",
    "Diagnose vanishing/exploding gradients in RNNs",
    "Implement basic RNN in PyTorch"
  ],
  "concepts": [
    "Sequential data: text, time series, speech",
    "Hidden state h_t as memory",
    "RNN cell: h_t = tanh(W_hh·h_{t-1} + W_xh·x_t + b)",
    "Unrolling through time",
    "Vanishing gradient in long sequences",
    "Exploding gradient and gradient clipping",
    "Teacher forcing in sequence training",
    "Bidirectional RNNs"
  ],
  "formulas": [
    {"name": "RNN hidden state", "latex": "h_t = \\tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)", "note": "Recurrent update at each timestep."},
    {"name": "Output", "latex": "y_t = W_{hy} h_t + b_y", "note": "Readout from hidden state."},
    {"name": "BPTT gradient", "latex": "\\frac{\\partial L}{\\partial W} = \\sum_t \\frac{\\partial L}{\\partial h_t} \\frac{\\partial h_t}{\\partial W}", "note": "Backprop through time — chain through all timesteps."}
  ],
  "references": [
    "Goodfellow et al. ch. 10",
    "Elman 1990 RNN paper",
    "Graves 'Supervised Sequence Labelling with RNNs'"
  ]
}
---

## Penjelasan

RNN adalah arsitektur untuk data sekuensial — teks, time series, speech, DNA. Berbeda dengan feedforward network yang memproses input secara independen, RNN memiliki *hidden state* yang membawa informasi dari timestep sebelumnya.

**Hidden state.** Pada setiap timestep t, RNN menghitung h_t = tanh(W_hh·h_{t-1} + W_xh·x_t + b). h_t merangkum semua informasi dari x_1...x_t — semacam "memori" yang di-update rekursif. Output y_t = W_hy·h_t + b_y.

**Unrolling.** Untuk training, RNN di-unroll melalui waktu: network dengan shared weights di setiap timestep, membentuk deep network dengan depth = sequence length. Backpropagation through time (BPTT) menghitung gradient dengan chain rule melalui semua timestep — gradient bisa sangat panjang.

**Masalah RNN klasik.**
- *Vanishing gradient*: gradient harus dikalikan W_hh berkali-kali; jika ||W_hh||<1, gradient -> 0 untuk dependensi jangka panjang. RNN "lupa" informasi awal.
- *Exploding gradient*: jika ||W_hh||>1, gradient -> ∞; mitigasi: gradient clipping.
- *Long-term dependencies*: sulit mempelajari hubungan antara x_1 dan x_100.

**Solusi.** LSTM dan GRU (modul berikutnya) menggunakan gating mechanisms untuk mengontrol informasi yang disimpan/dilupakan — mengatasi vanishing gradient. Attention mechanism (Transformer) menghilangkan rekursi sepenuhnya.

**Bidirectional RNN.** Dua RNN: satu forward, satu backward; output digabung. Berguna ketika konteks masa depan relevan (mis. named entity recognition).

## Konsep Penting

1. **Sequential != parallel** — RNN memproses timestep satu per satu; tidak parallelizable seperti CNN/Transformer.
2. **Shared weights** — W_hh dan W_xh sama di semua timestep; ini yang membuat RNN parameter-efficient.
3. **Hidden state ≠ output** — h_t adalah memori internal; y_t adalah prediksi untuk timestep t.
4. **BPTT memory cost** — menyimpan semua h_t untuk backward; sequence panjang = memory besar.
5. **Teacher forcing** — saat training sequence-to-sequence, input timestep berikutnya adalah ground truth (bukan prediksi sendiri) untuk stabilitas.

## Formula

- h_t = tanh(W_hh h_{t-1} + W_xh x_t + b_h).
- y_t = W_hy h_t + b_y.
- BPTT: ∂L/∂W_hh = Σ_t (∂L/∂h_t)(∂h_t/∂h_{t-1})...(∂h_{t-k}/∂W_hh).

## Contoh Kasus

Sentiment analysis pada review: RNN memproses kata per kata, hidden state mengakumulasi makna; output akhir = probabilitas sentimen. Masalah: review panjang -> RNN lupa awal kalimat. LSTM/attention menyelesaikan ini.

## Implementasi

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, h_n = self.rnn(x)
        # use last hidden state for classification
        return self.fc(h_n[-1])

# character-level language model
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        x = self.embed(x)
        out, h = self.rnn(x, h)
        return self.fc(out), h
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "RNN's hidden state serves as:",
    "options": [
      "A memory that carries information across timesteps",
      "The final output",
      "A fixed-size representation of input",
      "The loss function"
    ],
    "answer": 0,
    "explanation": "h_t accumulates information from all previous timesteps — the recurrent memory.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Vanishing gradient in RNNs occurs because:",
    "options": [
      "Gradients must propagate through many timesteps, multiplying by W_hh each step",
      "The network is too shallow",
      "Learning rate is too high",
      "Activation functions are missing"
    ],
    "answer": 0,
    "explanation": "BPTT multiplies gradients through all timesteps; if ||W_hh||<1, gradient vanishes for long-range dependencies.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "math",
    "question": "For h_t = tanh(W h_{t-1} + x_t), if W=0.5 and sequence length is 100, gradient magnitude roughly scales as:",
    "options": ["0.5^100 (vanishes)", "0.5^50", "100*0.5", "constant"],
    "answer": 0,
    "explanation": "Each timestep multiplies gradient by ~0.5*tanh' -> exponential decay.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `rnn_step(x_t, h_prev, W_xh, W_hh, b)` returning next hidden state using tanh activation.",
    "starter_code": "import numpy as np\n\ndef rnn_step(x_t, h_prev, W_xh, W_hh, b):\n    x_t = np.asarray(x_t, dtype=float)\n    h_prev = np.asarray(h_prev, dtype=float)\n    pass\n",
    "function": "rnn_step",
    "tests": [
      {"call": "rnn_step([1,0],[0,0],[[1,0],[0,1]],[[0.5,0],[0,0.5]],[0,0]).tolist()", "expected": [0.46211715726000974,0.0], "tolerance": 0.01}
    ],
    "hints": ["h = tanh(W_xh@x_t + W_hh@h_prev + b)", "check matrix multiplication orientation"],
    "explanation": "Single RNN timestep computation.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Why are Transformers (attention) preferred over RNNs for long sequences despite RNNs being 'natural' for sequential data? Discuss parallelization, long-range dependencies, and the role of positional encoding.",
    "rubric": [
      "RNNs process sequentially (slow training); Transformers parallelize over sequence",
          "Attention provides direct access to all positions (no vanishing gradient)",
      "Positional encoding injects order information that self-attention lacks",
      "RNNs still useful for streaming/low-latency; Transformers dominate for most NLP"
        ],
    "points": 2
  }
]
```

## Referensi

- Goodfellow et al. ch. 10 (sequence modeling)
- Elman (1990) 'Finding structure in time'
- Graves, 'Supervised Sequence Labelling with Recurrent Neural Networks'
- PyTorch nn.RNN, nn.LSTM, nn.GRU documentation
