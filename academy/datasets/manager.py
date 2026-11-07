"""Dataset registry: metadata, sources, licenses, fetch/generate logic."""
from __future__ import annotations

import csv
import json
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import BUNDLED_DIR, DOWNLOADED_DIR


@dataclass
class DatasetSpec:
    name: str
    stage: str            # beginner | intermediate | advanced | research
    description: str
    source: str           # url or "synthetic" or "bundled"
    license: str
    license_url: str = ""
    task: str = ""        # suggested ML task
    n_samples: int = 0
    n_features: int = 0
    target: str = ""
    download: bool = False  # True = fetched on demand; False = bundled/generated
    generator: str = ""     # name of synthetic generator function
    notes: str = ""
    extra: dict = field(default_factory=dict)


DATASETS: dict[str, DatasetSpec] = {}


def register(spec: DatasetSpec):
    DATASETS[spec.name] = spec


# ---------- built-in small datasets (bundled via sklearn at setup) ----------
register(DatasetSpec(
    name="iris", stage="beginner",
    description="Fisher's Iris flower dataset: 150 samples, 4 features, 3 species.",
    source="https://archive.ics.uci.edu/ml/datasets/iris",
    license="CC BY 4.0 (UCI ML Repository)",
    task="multiclass classification", n_samples=150, n_features=4, target="species",
    download=False, notes="Bundled locally as CSV for offline use.",
))
register(DatasetSpec(
    name="wine", stage="beginner",
    description="Wine recognition dataset: 178 samples, 13 chemical features, 3 cultivars.",
    source="https://archive.ics.uci.edu/ml/datasets/Wine",
    license="CC BY 4.0 (UCI ML Repository)",
    task="multiclass classification", n_samples=178, n_features=13, target="cultivar",
    download=False,
))
register(DatasetSpec(
    name="breast_cancer", stage="beginner",
    description="Breast Cancer Wisconsin (Diagnostic): 569 samples, 30 features.",
    source="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)",
    license="Public domain (UCI ML Repository)",
    task="binary classification", n_samples=569, n_features=30, target="diagnosis",
    download=False,
))
register(DatasetSpec(
    name="california_housing", stage="beginner",
    description="California housing prices: ~20k samples, 8 features, median house value target.",
    source="https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset",
    license="Open (sklearn bundled, derived from 1990 US Census)",
    task="regression", n_samples=20640, n_features=8, target="MedHouseVal",
    download=False,
))
register(DatasetSpec(
    name="titanic", stage="intermediate",
    description="Titanic passenger survival dataset (~891 rows) with missing values and mixed types.",
    source="https://www.kaggle.com/c/titanic (via seaborn)",
    license="Kaggle competition data / seaborn mirror",
    task="binary classification + feature engineering", target="survived",
    download=False, notes="Fetched once from seaborn's GitHub mirror and cached locally.",
))
register(DatasetSpec(
    name="digits", stage="intermediate",
    description="Handwritten digits 0-9: 1797 samples of 8x8 grayscale images.",
    source="https://scikit-learn.org/stable/datasets/overview.html",
    license="BSD-3 (bundled with sklearn)",
    task="image classification", n_samples=1797, target="digit",
    download=False,
))
register(DatasetSpec(
    name="fashion_mnist", stage="intermediate",
    description="Fashion-MNIST: 70k 28x28 grayscale images of clothing categories.",
    source="https://github.com/zalandoresearch/fashion-mnist",
    license="MIT", task="image classification", n_samples=70000,
    download=True, notes="Downloads IDX files on first use; cache in data/downloaded/.",
))
register(DatasetSpec(
    name="mnist", stage="intermediate",
    description="MNIST handwritten digits: 70k 28x28 grayscale images.",
    source="http://yann.lecun.com/exdb/mnist/",
    license="CC BY-SA 3.0", task="image classification", n_samples=70000,
    download=True,
))
register(DatasetSpec(
    name="credit_card_fraud", stage="intermediate",
    description="Credit card fraud detection: ~284k transactions, 30 PCA-transformed features, heavy imbalance.",
    source="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
    license="CC0 / Open (Kaggle, ULB)",
    task="imbalanced binary classification", n_samples=284807, target="Class",
    download=False, notes="Requires Kaggle credentials for the official file; a synthetic "
        "public-domain alternative generator is provided offline (see synthetic_fraud).",
        generator="synthetic_fraud",
))
register(DatasetSpec(
    name="cifar10", stage="advanced",
    description="CIFAR-10: 60k 32x32 color images in 10 classes.",
    source="https://www.cs.toronto.edu/~kriz/cifar.html",
    license="MIT-like (research use)", task="image classification", n_samples=60000,
    download=True,
))
register(DatasetSpec(
    name="imdb", stage="advanced",
    description="IMDB movie review sentiment: 50k labeled reviews.",
    source="https://ai.stanford.edu/~amaas/data/sentiment/",
    license="Research use (Stanford)", task="text classification", n_samples=50000,
    download=True,
))
register(DatasetSpec(
    name="synthetic_classification", stage="beginner",
    description="Synthetic 2-blob/ moons classification data generated locally with sklearn.",
    source="synthetic", license="Generated locally (no license restrictions)",
    task="classification", generator="synthetic_classification",
))
register(DatasetSpec(
    name="synthetic_regression", stage="beginner",
    description="Synthetic regression data with controllable noise and nonlinearity.",
    source="synthetic", license="Generated locally", task="regression",
    generator="synthetic_regression",
))
register(DatasetSpec(
    name="synthetic_fraud", stage="intermediate",
    description="Synthetic credit-card-fraud-style data: legitimate transactions ~ Gaussian, "
        "fraud ~ different mean/variance mixture. For offline practice only.",
    source="synthetic", license="Generated locally", task="imbalanced classification",
    generator="synthetic_fraud", n_samples=100000,
))
register(DatasetSpec(
    name="synthetic_timeseries", stage="advanced",
    description="Synthetic time series: trend + seasonality + noise + regime shifts.",
    source="synthetic", license="Generated locally", task="forecasting",
    generator="synthetic_timeseries",
))
register(DatasetSpec(
    name="synthetic_gp", stage="advanced",
    description="Synthetic dataset for symbolic regression / genetic programming: "
        "y = a*x1^2 + b*sin(x2) + noise.",
    source="synthetic", license="Generated locally", task="symbolic regression",
    generator="synthetic_gp", n_features=2,
))
register(DatasetSpec(
    name="nasa_exoplanet_archive", stage="research",
    description="NASA Exoplanet Archive: confirmed exoplanets with physical parameters.",
    source="https://exoplanetarchive.ipac.caltech.edu/",
    license="NASA open data (public domain)",
    task="regression/classification research", n_samples=5500,
    download=True,
))
register(DatasetSpec(
    name="kepler_koi", stage="research",
    description="Kepler Objects of Interest: transit metrics for ~9k planet candidates.",
    source="https://exoplanetarchive.ipac.caltech.edu/ (KOI table)",
    license="NASA open data", task="classification research",
    download=True,
))
register(DatasetSpec(
    name="nasa_near_earth_objects", stage="research",
    description="NASA Near-Earth Objects: close-approach asteroid data.",
    source="https://cneos.jpl.nasa.gov/",
    license="NASA/JPL open data", task="time-series/anomaly research",
    download=True,
))


# ---------- generators ----------

def _gen_synthetic_classification(seed=42, n=2000):
    from sklearn.datasets import make_moons, make_blobs
    n1, n2 = n // 2, n - n // 2
    X1, y1 = make_blobs(n_samples=n1, centers=[[0, 0], [4, 4]],
                        cluster_std=0.8, random_state=seed)
    X2, y2 = make_moons(n_samples=n2, noise=0.15, random_state=seed + 1)
    X2 = X2 + 2.0  # shift moons away from blobs
    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2 + 2])  # labels 0..3
    n_feat = X.shape[1]
    cols = [f"f{i+1}" for i in range(n_feat)] + ["target"]
    return pd.DataFrame(np.hstack([X, y.reshape(-1, 1)]), columns=cols)


def _gen_synthetic_regression(seed=42, n=2000):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n)
    y = 2 * x**2 - 1.5 * x + np.sin(2 * x) + rng.normal(0, 0.5, n)
    return pd.DataFrame({"x": x, "noise_feat": rng.normal(0, 1, n), "y": y})


def _gen_synthetic_fraud(seed=42, n=100000, fraud_ratio=0.0017):
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_ratio)
    n_legit = n - n_fraud
    legit = rng.normal(0.0, 1.0, (n_legit, 28))
    # fraud: shifted mean + different covariance signature
    fraud = rng.normal(2.5, 1.8, (n_fraud, 28))
    X = np.vstack([legit, fraud])
    y = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)]).astype(int)
    cols = [f"V{i+1}" for i in range(28)]
    df = pd.DataFrame(X, columns=cols)
    df["Amount"] = rng.lognormal(3, 1, n)
    df["Class"] = y
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _gen_synthetic_timeseries(seed=42, days=730):
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    trend = 0.05 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 7) + 3 * np.sin(2 * np.pi * t / 365.25)
    noise = rng.normal(0, 1.5, days)
    regime = (t > days * 0.6) * (0.08 * (t - days * 0.6))
    y = 50 + trend + seasonal + noise + regime
    return pd.DataFrame({"date": pd.date_range("2023-01-01", periods=days),
                         "value": y, "t": t})


def _gen_synthetic_gp(seed=42, n=500):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2, 2, n)
    x2 = rng.uniform(-2, 2, n)
    y = 1.7 * x1**2 + 0.8 * np.sin(2.3 * x2) - 0.4 * x1 * x2 + rng.normal(0, 0.05, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


GENERATORS = {
    "synthetic_classification": _gen_synthetic_classification,
    "synthetic_regression": _gen_synthetic_regression,
    "synthetic_fraud": _gen_synthetic_fraud,
    "synthetic_timeseries": _gen_synthetic_timeseries,
    "synthetic_gp": _gen_synthetic_gp,
}


# ---------- loading ----------

def _bundled_path(name: str) -> Path:
    return BUNDLED_DIR / f"{name}.csv"


def _load_bundled(name: str) -> pd.DataFrame:
    path = _bundled_path(name)
    if not path.exists():
        # build it once via sklearn/seaborn
        build_bundled_dataset(name)
    return pd.read_csv(path)


def build_bundled_dataset(name: str):
    """Create bundled CSVs using sklearn/seaborn (run once at setup)."""
    from sklearn import datasets as skd

    if name == "iris":
        d = skd.load_iris(as_frame=True)
        df = d.frame.rename(columns={"target": "species"})
    elif name == "wine":
        d = skd.load_wine(as_frame=True)
        df = d.frame.rename(columns={"target": "cultivar"})
    elif name == "breast_cancer":
        d = skd.load_breast_cancer(as_frame=True)
        df = d.frame.rename(columns={"target": "diagnosis"})
    elif name == "california_housing":
        d = skd.fetch_california_housing(as_frame=True)
        df = d.frame
    elif name == "digits":
        d = skd.load_digits()
        df = pd.DataFrame(d.data, columns=[f"pixel_{i}" for i in range(64)])
        df["digit"] = d.target
    elif name == "titanic":
        import seaborn as sns
        df = sns.load_dataset("titanic")
    elif name == "credit_card_fraud":
        # synthetic fallback if Kaggle file unavailable
        df = GENERATORS["synthetic_fraud"]()
    else:
        raise KeyError(f"No bundled builder for {name}")
    df.to_csv(_bundled_path(name), index=False)
    return df


def load_dataset(name: str, **kwargs) -> pd.DataFrame:
    spec = DATASETS[name]
    if spec.generator:
        return GENERATORS[spec.generator](**kwargs)
    if not spec.download:
        return _load_bundled(name)
    # on-demand downloads
    path = DOWNLOADED_DIR / f"{name}"
    path.mkdir(parents=True, exist_ok=True)
    if name == "mnist":
        return _download_mnist(path, **kwargs)
    if name == "fashion_mnist":
        return _download_fashion_mnist(path, **kwargs)
    if name == "cifar10":
        return _download_cifar10(path, **kwargs)
    if name == "imdb":
        return _download_imdb(path, **kwargs)
    if name in ("nasa_exoplanet_archive", "kepler_koi", "nasa_near_earth_objects"):
        return _download_nasa(name, path, **kwargs)
    raise KeyError(f"No downloader for {name}")


def _download(url: str, dest: Path):
    if dest.exists():
        return
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)


# --- specific downloaders (best-effort, cache locally) ---

def _download_mnist(path: Path, **kw):
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    for f in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
              "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
        _download(base + f, path / f)
    # parse IDX
    import gzip, struct
    def read_images(fp):
        with gzip.open(fp, "rb") as f:
            magic, n, r, c = struct.unpack(">IIII", f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r * c)
    def read_labels(fp):
        with gzip.open(fp, "rb") as f:
            magic, n = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)
    Xtr = read_images(path / "train-images-idx3-ubyte.gz")
    ytr = read_labels(path / "train-labels-idx1-ubyte.gz")
    Xte = read_images(path / "t10k-images-idx3-ubyte.gz")
    yte = read_labels(path / "t10k-labels-idx1-ubyte.gz")
    return {"X_train": Xtr, "y_train": ytr, "X_test": Xte, "y_test": yte}


def _download_fashion_mnist(path: Path, **kw):
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    # fashion-mnist uses different labels; fetch from github mirror
    base2 = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
    mapping = {
        "train_images": ("train-images-idx3-ubyte.gz", base2 + "train-images-idx3-ubyte.gz"),
        "train_labels": ("train-labels-idx1-ubyte.gz", base2 + "train-labels-idx1-ubyte.gz"),
        "test_images": ("t10k-images-idx3-ubyte.gz", base2 + "t10k-images-idx3-ubyte.gz"),
        "test_labels": ("t10k-labels-idx1-ubyte.gz", base2 + "t10k-labels-idx1-ubyte.gz"),
    }
    import gzip, struct
    def read_images(fp):
        with gzip.open(fp, "rb") as f:
            magic, n, r, c = struct.unpack(">IIII", f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r * c)
    def read_labels(fp):
        with gzip.open(fp, "rb") as f:
            magic, n = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)
    out = {}
    for key, (fname, url) in mapping.items():
        dest = path / fname
        _download(url, dest)
        out[key] = read_labels(dest) if "labels" in key else read_images(dest)
    return {"X_train": out["train_images"], "y_train": out["train_labels"],
            "X_test": out["test_images"], "y_test": out["test_labels"]}


def _download_cifar10(path: Path, **kw):
    import pickle, tarfile, urllib.request as ur
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tgz = path / "cifar-10-python.tar.gz"
    _download(url, tgz)
    if not (path / "cifar-10-batches-py").exists():
        with tarfile.open(tgz) as tf:
            tf.extractall(path)
    def load_batch(fp):
        with open(fp, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return d[b"data"].reshape(-1, 3, 32, 32), np.array(d[b"labels"])
    Xs, ys = [], []
    for i in range(1, 6):
        X, y = load_batch(path / "cifar-10-batches-py" / f"data_batch_{i}")
        Xs.append(X); ys.append(y)
    try:
        Xte, yte = load_batch(path / "cifar-10-batches-py" / "test_batch")
    except Exception:
        Xte = yte = None
    return {"X_train": np.concatenate(Xs), "y_train": np.concatenate(ys),
            "X_test": Xte, "y_test": yte}


def _download_imdb(path: Path, **kw):
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tgz = path / "aclImdb_v1.tar.gz"
    _download(url, tgz)
    import tarfile
    root = path / "aclImdb"
    if not root.exists():
        with tarfile.open(tgz) as tf:
            tf.extractall(path)
    rows = []
    for split in ["train", "test"]:
        for label, name in [("pos", 1), ("neg", 0)]:
            d = root / split / label
            for f in d.glob("*.txt"):
                rows.append((f.read_text(encoding="utf-8", errors="ignore"), name))
    return pd.DataFrame(rows, columns=["review", "sentiment"])


def _download_nasa(name: str, path: Path, **kw):
    table = {
        "nasa_exoplanet_archive": "ps",
        "kepler_koi": "koi",
        "nasa_near_earth_objects": None,
    }[name]
    if table is None:
        # CNEOS close-approach API sample
        url = "https://ssd-api.jpl.nasa.gov/cad.api?date-min=2024-01-01&date-max=2024-12-31&dist-max=0.05"
        data = json.loads(urllib.request.urlopen(url, timeout=60).read())
        fields = data["fields"]
        rows = [dict(zip(fields, r)) for r in data["data"]]
        return pd.DataFrame(rows)
    url = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
           f"?query=select+*&format=csv&table=PSCOMPPARS" if table == "ps"
           else "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
                "?query=select+*&format=csv&table=KOI")
    dest = path / f"{name}.csv"
    _download(url, dest)
    return pd.read_csv(dest)


def list_datasets() -> list[DatasetSpec]:
    return list(DATASETS.values())


def dataset_info(name: str) -> DatasetSpec:
    return DATASETS[name]
