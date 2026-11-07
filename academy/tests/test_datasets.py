"""Tests for the dataset registry and generators."""
import pandas as pd
import pytest

from academy.datasets.manager import (
    DATASETS, list_datasets, dataset_info, load_dataset,
)


def test_registry_size_and_stages():
    specs = list_datasets()
    assert len(specs) >= 15
    stages = {s.stage for s in specs}
    assert {"beginner", "intermediate", "advanced", "research"} <= stages


def test_metadata_completeness():
    for s in list_datasets():
        assert s.name and s.description and s.source and s.license
        assert s.task, f"{s.name} missing task"


def test_synthetic_generators():
    df = load_dataset("synthetic_regression", seed=1, n=100)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["x", "noise_feat", "y"]
    assert len(df) == 100

    df = load_dataset("synthetic_gp", seed=1, n=50)
    assert set(df.columns) == {"x1", "x2", "y"}
    # sanity: y should correlate with x1^2
    assert df["y"].std() > 0

    df = load_dataset("synthetic_fraud", seed=1, n=5000)
    assert "Class" in df.columns
    assert df["Class"].mean() < 0.05  # heavy imbalance

    df = load_dataset("synthetic_timeseries", seed=1, days=100)
    assert len(df) == 100 and "value" in df.columns


def test_synthetic_classification_has_target():
    df = load_dataset("synthetic_classification", seed=1, n=200)
    assert "target" in df.columns
    assert df["target"].nunique() >= 2


def test_bundled_iris_loads():
    df = load_dataset("iris")
    assert len(df) == 150
    assert "species" in df.columns


def test_download_flag_consistency():
    for name, spec in DATASETS.items():
        if spec.download:
            assert spec.generator == "", f"{name} both download and generator"
        if spec.generator:
            assert spec.generator in __import__(
                "academy.datasets.manager", fromlist=["GENERATORS"]).GENERATORS


def test_dataset_info():
    s = dataset_info("iris")
    assert s.n_samples == 150
