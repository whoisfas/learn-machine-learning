---
{
  "id": "p5-reproducible-research",
  "title": "Reproducible Research",
  "phase": 5,
  "difficulty": "intermediate",
  "estimated_hours": 10,
  "objectives": [
    "Understand reproducibility crisis in ML",
    "Implement reproducible experiments: seeds, versioning, documentation",
    "Use tools: DVC, MLflow, Weights & Biases, Docker",
    "Create research artifacts that others can verify"
  ],
  "concepts": [
    "Reproducibility vs replicability vs repeatability",
    "Random seed management",
    "Dependency management: requirements.txt, environment.yml, Docker",
    "Data versioning: DVC, Git LFS",
    "Experiment tracking: MLflow, W&B, Neptune",
    "Code organization for research",
    "Documentation standards for papers",
    "Artifact evaluation badges (ACM, NeurIPS)"
  ],
  "formulas": [],
  "references": [
    "Pineau et al. 'Improving Reproducibility in ML Research' (NeurIPS 2021)",
    "ACM Artifact Review and Badging",
    "MLflow documentation",
    "DVC documentation",
    "'The Reproducibility Crisis in ML' — various"
  ]
}
---

## Penjelasan

Reproducibility adalah kemampuan untuk mendapatkan hasil yang sama dari eksperimen yang sama — fondasi scientific credibility. ML research menghadapi *reproducibility crisis*: banyak paper tidak bisa direproduksi karena code/data tidak tersedia, seed tidak dicatat, atau environment tidak terdokumentasi.

**Levels of reproducibility.**
- *Repeatability*: same team, same setup -> same results (weakest).
- *Reproducibility*: different team, same method + data -> similar results.
- *Replicability*: different method/data -> same conclusion (strongest claim).

**Sources of irreproducibility.**
- *Randomness*: tidak dicatat seed untuk numpy/torch/sklearn.
- *Dependencies*: versi library berbeda -> hasil berbeda.
- *Data*: data preprocessing tidak terdokumentasi atau data tidak tersedia.
- *Hardware*: GPU non-determinism, floating point differences.
- *Code*: tidak ada code release, atau code tidak runnable.

**Best practices.**
- *Seed everything*: `random.seed()`, `np.random.seed()`, `torch.manual_seed()`, `torch.backends.cudnn.deterministic=True`.
- *Pin dependencies*: `requirements.txt` dengan exact versions atau `environment.yml`.
- *Version data*: DVC (Data Version Control) untuk large files; atau document download source + checksum.
- *Track experiments*: MLflow/W&B untuk logging params, metrics, artifacts.
- *Code organization*: modular code, README dengan run instructions, config files untuk hyperparameters.
- *Docker*: containerize environment untuk exact reproduction.

## Konsep Penting

1. **Seed is necessary but not sufficient** — GPU non-determinism tetap bisa menyebabkan perbedaan kecil.
2. **Document everything** — preprocessing steps, hyperparameters, hardware, library versions.
3. **Reproducibility checklist** — sebelum submit paper: code available? data available? seed documented? environment specified?
4. **Artifact badges** — ACM/NeurIPS memberikan badges untuk reproducible research; increasing prestige.
5. **Reproducibility is a spectrum** — perfect bit-for-bit reproduction mungkin tidak mungkin; aim for statistical equivalence.

## Formula

- Reproducibility checklist: seed + data + code + environment + hardware.
- Statistical equivalence: results within confidence intervals across runs.

## Contoh Kasus

Paper NeurIPS dengan reproducibility badge: code di GitHub, data di Zenodo, Dockerfile untuk environment, seeds documented, experiment configs dalam YAML, results reproduced by independent team within 1% variance.

## Implementasi

```python
# Reproducibility utilities
import random, os, json, hashlib
import numpy as np
import torch

def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_experiment_config(config, path='config.json'):
    """Save experiment configuration."""
    with open(path, 'w') as f:
        json.dump(config, f, indent=2, sort_keys=True)
    # compute hash for verification
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

# MLflow experiment tracking
import mlflow

def track_experiment(params, metrics, artifacts=None):
    """Log experiment to MLflow."""
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if artifacts:
            for name, path in artifacts.items():
                mlflow.log_artifact(path, name)
        run_id = mlflow.active_run().info.run_id
        return run_id

# requirements.txt generation
import subprocess
def freeze_requirements(path='requirements.txt'):
    subprocess.run(['pip', 'freeze'], stdout=open(path, 'w'))
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Reproducibility in ML research means:",
    "options": [
      "Independent researchers can obtain similar results using documented methods",
      "Getting bit-identical results every time",
      "Publishing code only",
      "Using the same hardware"
    ],
    "answer": 0,
    "explanation": "Reproducibility = same method + data -> similar results, not necessarily identical.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "The most common cause of irreproducible ML results is:",
    "options": [
      "Not documenting seeds, dependencies, and preprocessing",
      "Using GPUs",
      "Having too many authors",
      "Publishing in top venues"
    ],
    "answer": 0,
    "explanation": "Missing documentation of randomness, versions, and data processing.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "DVC (Data Version Control) is used for:",
    "options": [
      "Versioning large datasets alongside code",
      "Training models",
      "Visualizing data",
      "Writing papers"
    ],
    "answer": 0,
    "explanation": "DVC tracks large files in Git without storing them in repo.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `reproducibility_checklist()` returning dict of items to verify before submitting research.",
    "starter_code": "def reproducibility_checklist():\n    pass\n",
    "function": "reproducibility_checklist",
    "tests": [
      {"call": "len(reproducibility_checklist()) >= 8", "expected": true}
    ],
    "hints": ["Include: seeds, requirements, data source, preprocessing, configs, results, code availability, hardware"],
    "explanation": "Comprehensive reproducibility checklist.",
    "points": 1
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "You try to reproduce a famous ML paper but get 10% lower accuracy. List five possible causes and how you'd investigate each.",
    "rubric": [
      "Different library versions -> check requirements, use Docker",
          "Missing preprocessing details -> check paper supplement, contact authors",
      "Different random seeds -> run multiple seeds, check variance",
      "Data version differences -> verify dataset checksum/version",
      "Hardware differences -> GPU vs CPU, deterministic flags"
        ],
    "points": 2
  }
]
```

## Referensi

- Pineau et al. (2021) 'Improving Reproducibility in ML Research' — NeurIPS
- ACM Artifact Review and Badging
- MLflow documentation (https://mlflow.org)
- DVC documentation (https://dvc.org)
- 'The Reproducibility Crisis in ML' — special issues in JMLR, NeurIPS
