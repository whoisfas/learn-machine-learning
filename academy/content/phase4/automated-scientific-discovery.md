---
{
  "id": "p4-automated-scientific-discovery",
  "title": "Automated Scientific Discovery",
  "phase": 4,
  "difficulty": "advanced",
  "estimated_hours": 14,
  "objectives": [
    "Understand the pipeline from hypothesis to discovery",
    "Apply ML to scientific domains: physics, chemistry, biology",
    "Design experiments that can falsify hypotheses",
    "Connect automated discovery to reproducible research"
  ],
  "concepts": [
    "Scientific method: hypothesis -> experiment -> analysis -> conclusion",
    "Automated hypothesis generation",
    "Active learning for experiment design",
    "Symbolic regression for law discovery",
    "Surrogate models for expensive simulations",
    "Bayesian optimization for materials/chemistry",
    "Self-driving labs",
    "Reproducibility in automated science"
  ],
  "formulas": [],
  "references": [
    "Schmidt & Lipson (2009) 'Distilling Free-Form Natural Laws'",
    "Udrescu & Tegmark (2020) 'AI Feynman'",
    "Coley et al. 'Robotic Discovery in Chemistry'",
    "Nature Machine Intelligence automated discovery papers"
  ]
}
---

## Penjelasan

Automated Scientific Discovery menggunakan ML untuk mempercepat atau mengotomatisasi proses penemuan ilmiah — dari hipotesis hingga eksperimen hingga analisis. Ini adalah frontier research yang menggabungkan ML dengan domain science.

**Paradigma discovery.** Proses ilmiah tradisional: observasi -> hipotesis -> eksperimen -> analisis -> kesimpulan. Otomatisasi bisa di setiap tahap: hipotesis generation (ML dari literature), eksperimen design (active learning, BO), analisis (SR, anomaly detection), dan bahkan eksekusi (self-driving labs dengan robot).

**Contoh nyata.**
- *Fisik*: AI Feynman mereproduksi hukum fisika (gravitasi, relativitas) dari data orbit — tanpa diberi tahu bentuk fungsinya.
- *Kimia*: self-driving labs mensintesis dan menguji ribuan kandidat material per hari; ML memprediksi reaktivitas untuk memandu eksperimen berikutnya.
- *Biologi*: AlphaFold memprediksi struktur protein dari sekuens — salah satu terobosan ML terbesar dalam biologi.
- *Astronomi*: ML menemukan exoplanet candidates dari transit data (Kepler, TESS).

**Active learning.** Alih-alih random sampling, ML memilih eksperimen yang paling informatif — eksperimen yang expected-nya paling mengurangi ketidakpastian model. BO untuk materials discovery adalah contoh: pilih kandidat material yang expected improvement tertinggi.

**Reproducibility.** Automated discovery menghasilkan ribuan eksperimen; dokumentasi yang rapi (seed, kode, data, environment) adalah syarat untuk reproducibility. Setiap eksperimen harus tercatat: parameter, hasil, metrik, artefak.

## Konsep Penting

1. **Discovery != prediction** — model yang akurat belum tentu menemukan hukum baru; butuh interpretasi dan validasi ilmiah.
2. **Hypothesis space matters** — SR/GP hanya bisa menemukan fungsi dalam primitive set; domain knowledge crucial.
3. **Validation adalah raja** — eksperimen yang bisa memfalsifikasi hipotesis lebih berharga dari eksperimen yang mengkonfirmasi.
4. **Self-driving labs** — robot + ML loop menutup siklus eksperimen; mahal tapi sangat powerful untuk chemistry/materials.
5. **Human-in-the-loop** — fully automated discovery masih langka; human expertise tetap diperlukan untuk interpretasi dan arah.

## Formula

- Active learning: pilih x yang memaksimalkan uncertainty atau expected information gain.
- BO for discovery: acquisition = expected improvement over current best.

## Contoh Kasus

Baterai materials discovery: 10,000 kandidat komposisi; ML model memprediksi kapasitas dari komposisi; BO memilih 100 kandidat paling promising untuk disintesis dan diuji; update model; iterasi. Hasil: menemukan material dengan kapasitas 20% lebih tinggi dari baseline dalam 6 bulan vs 2 tahun dengan trial-and-error.

## Implementasi

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

def active_discovery_loop(objective_fn, bounds, n_iter=50, n_init=10):
    """BO loop for scientific discovery."""
    rng = np.random.default_rng(42)
    # initial random exploration
    X = rng.uniform(bounds[:,0], bounds[:,1], (n_init, bounds.shape[0]))
    y = np.array([objective_fn(x) for x in X])
    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(bounds.shape[0]))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    for i in range(n_iter):
        gp.fit(X, y)
        # acquisition: UCB
        X_cand = rng.uniform(bounds[:,0], bounds[:,1], (500, bounds.shape[0]))
        mu, sigma = gp.predict(X_cand, return_std=True)
        ucb = mu + 1.5 * sigma
        x_new = X_cand[np.argmax(ucb)]
        y_new = objective_fn(x_new)
        X = np.vstack([X, x_new])
        y = np.append(y, y_new)
        print(f"iter {i}: best={y.max():.4f}, candidate={x_new}")
    return X[np.argmax(y)], y.max()

# experiment logging for reproducibility
import json, hashlib
from datetime import datetime

def log_experiment(name, params, results, code_hash=None):
    record = {
        'name': name,
        'timestamp': datetime.now().isoformat(),
        'params': params,
        'results': results,
        'code_hash': code_hash or hashlib.md5(
            json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    }
    with open('experiments.jsonl', 'a') as f:
        f.write(json.dumps(record) + '\n')
    return record
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Automated scientific discovery differs from standard ML prediction because it:",
    "options": [
      "Aims to discover new laws/patterns, not just predict labels",
      "Uses more data",
      "Is faster",
      "Requires no domain knowledge"
    ],
    "answer": 0,
    "explanation": "Discovery = finding new knowledge; prediction = applying known patterns.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Active learning in scientific discovery means:",
    "options": [
      "ML selects the most informative experiments to run next",
      "Running experiments automatically",
      "Using all possible experiments",
      "Random experiment selection"
    ],
    "answer": 0,
    "explanation": "Active learning chooses experiments that maximally reduce model uncertainty.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "The main challenge in automated discovery is:",
    "options": [
      "Ensuring discovered patterns are real laws, not artifacts or overfitting",
      "Computational speed",
      "Data storage",
      "Model size"
    ],
    "answer": 0,
    "explanation": "Validation and falsification are crucial; automated systems can find spurious patterns.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `information_gain(X_pool, model, X_train, y_train)` estimating which pool point would most reduce model uncertainty (using prediction variance as proxy).",
    "starter_code": "import numpy as np\n\ndef information_gain(X_pool, model_predict_fn, X_train, y_train):\n    # model_predict_fn returns (mean, std) for each pool point\n    pass\n",
    "function": "information_gain",
    "tests": [
      {"call": "information_gain([[0.5]], lambda x: (np.array([0.5]), np.array([0.1])), None, None)", "expected": "check"}
    ],
    "hints": ["Return uncertainty (std) for each pool point; higher = more informative"],
    "explanation": "Uncertainty sampling for active learning.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Design a self-driving lab pipeline for discovering a new battery electrolyte. Describe the ML components, experiment loop, safety constraints, and how you'd validate discoveries before publication.",
    "rubric": [
      "ML model predicts electrolyte properties from composition",
          "BO/active learning selects next experiments",
      "Robot executes synthesis + testing automatically",
      "Safety constraints: human approval for dangerous reactions",
      "Validation: independent replication, peer review, ablation studies"
        ],
    "points": 3
  }
]
```

## Referensi

- Schmidt & Lipson (2009) 'Distilling Free-Form Natural Laws from Experimental Data'
- Udrescu & Tegmark (2020) 'AI Feynman'
- Coley et al. 'Robotic Discovery in Chemistry'
- 'Self-driving laboratories' — Nature Reviews Materials
- 'Machine learning for molecular and materials science' — Butler et al. 2018
