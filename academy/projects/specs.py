"""Project-based learning specs: 15 progressive projects."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProjectSpec:
    id: str
    title: str
    stage: str            # beginner | intermediate | advanced
    phase: int
    brief: str
    objectives: list
    dataset: str
    task_breakdown: list
    starter_code: str
    acceptance_criteria: list
    metrics: list
    deliverables: list
    research_extension: str
    hints: list = field(default_factory=list)


PROJECTS: list[ProjectSpec] = [
    ProjectSpec(
        id="p01-house-price",
        title="House Price Prediction",
        stage="beginner", phase=2,
        brief="Predict California median house values using regression. Learn the full ML pipeline: load, explore, preprocess, split, train, evaluate.",
        objectives=[
            "Load and explore the California Housing dataset",
            "Handle features and build a preprocessing pipeline",
            "Train LinearRegression and at least one nonlinear model",
            "Evaluate with RMSE and R^2 and compare models",
        ],
        dataset="california_housing",
        task_breakdown=[
            "EDA: distributions, correlations, missing values",
            "Preprocessing: scaling, optional feature engineering",
            "Baseline: LinearRegression",
            "Improvement: RandomForestRegressor",
            "Evaluation + residual analysis",
        ],
        starter_code='''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# TODO: load california_housing, split, train, evaluate
''',
        acceptance_criteria=[
            "Script runs end-to-end without errors",
            "LinearRegression RMSE reported",
            "At least one second model compared",
            "Train/test split used (no leakage)",
        ],
        metrics=["RMSE", "MAE", "R^2"],
        deliverables=["solution.py", "results.md with metrics table", "residual plot"],
        research_extension="Try GradientBoosting / HistGradientBoosting and tune with cross-validation.",
        hints=["Use fetch_california_housing(as_frame=True)", "Scale features before linear models"],
    ),
    ProjectSpec(
        id="p02-iris-classification",
        title="Iris Classification",
        stage="beginner", phase=2,
        brief="Classic multiclass classification on Iris. Focus on evaluation discipline: stratified split, confusion matrix, classification report.",
        objectives=["Train and compare 3 classifiers", "Produce confusion matrices",
                    "Explain which features matter most"],
        dataset="iris",
        task_breakdown=["Load iris", "Train/test split (stratify)", "KNN, LogisticRegression, DecisionTree",
                        "Compare accuracy/F1", "Feature importance analysis"],
        starter_code='''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# TODO: full pipeline
''',
        acceptance_criteria=["3 models trained", "classification_report printed for each",
                             "confusion_matrix visualized or printed"],
        metrics=["accuracy", "F1-macro", "confusion matrix"],
        deliverables=["solution.py", "model_comparison.md"],
        research_extension="Implement a simple grid search over k for KNN.",
    ),
    ProjectSpec(
        id="p03-student-performance",
        title="Student Performance Prediction",
        stage="beginner", phase=2,
        brief="Predict final exam score from study habits using the Student Performance dataset (generate synthetic version locally if needed).",
        objectives=["Regression on tabular data", "Feature engineering (interaction terms)",
                    "Regularization comparison"],
        dataset="synthetic_regression",
        task_breakdown=["EDA", "Feature engineering", "LinearRegression vs Ridge vs Lasso",
                        "Cross-validation comparison"],
        starter_code='''# Student performance: predict score from hours_studied, attendance, sleep, prior_gpa
# TODO: create or load dataset, engineer features, compare regressors
''',
        acceptance_criteria=["At least 2 regressors compared with CV", "Feature engineering documented",
                             "Train/validation/test split respected"],
        metrics=["RMSE", "R^2", "CV score"],
        deliverables=["solution.py", "feature_analysis.md"],
        research_extension="Fit a small neural network in PyTorch and compare.",
    ),
    ProjectSpec(
        id="p04-customer-segmentation",
        title="Customer Segmentation",
        stage="beginner", phase=2,
        brief="Unsupervised clustering of synthetic customer data into behavioral segments.",
        objectives=["Apply K-Means", "Choose k with elbow + silhouette", "Interpret clusters"],
        dataset="synthetic_classification",
        task_breakdown=["Generate/load customer-like data", "Scale features", "K-Means for k=2..10",
                        "Elbow + silhouette analysis", "Cluster profiling"],
        starter_code='''from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# TODO: clustering pipeline
''',
        acceptance_criteria=["Silhouette score reported for chosen k", "Cluster profiles described",
                             "K-Means run for a range of k values"],
        metrics=["inertia", "silhouette score"],
        deliverables=["solution.py", "segmentation_report.md"],
        research_extension="Compare with GaussianMixture and DBSCAN.",
    ),
    ProjectSpec(
        id="p05-data-visualization",
        title="Exploratory Data Visualization",
        stage="beginner", phase=1,
        brief="Build a publication-quality EDA notebook/report for the Titanic dataset: univariate, bivariate, missingness.",
        objectives=["Use matplotlib/seaborn effectively", "Tell a data story", "Handle missing data visually"],
        dataset="titanic",
        task_breakdown=["Load + clean", "Univariate plots", "Bivariate (survival vs features)",
                        "Missingness analysis", "Written insights"],
        starter_code='''import seaborn as sns
import matplotlib.pyplot as plt
# TODO: EDA of titanic dataset
''',
        acceptance_criteria=[">=6 distinct plot types", "Missingness quantified", "Insights written"],
        metrics=["n/a (qualitative rubric)"],
        deliverables=["eda.ipynb or eda.py + plots/", "insights.md"],
        research_extension="Add interactive plots with plotly (optional).",
    ),
    ProjectSpec(
        id="p06-fraud-detection",
        title="Fraud Detection",
        stage="intermediate", phase=2,
        brief="Imbalanced binary classification on credit-card-fraud-style data. Handle imbalance, choose threshold, interpret PR-AUC.",
        objectives=["Deal with class imbalance", "Use PR-AUC not just accuracy", "Threshold tuning"],
        dataset="credit_card_fraud",
        task_breakdown=["EDA of imbalance", "Baseline classifier", "Resampling/scale_pos_weight",
                        "Threshold optimization on validation", "Final test evaluation"],
        starter_code='''from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
# TODO: fraud detection pipeline
''',
        acceptance_criteria=["PR-AUC reported", "Threshold justified", "No test-set leakage in tuning"],
        metrics=["PR-AUC", "ROC-AUC", "precision/recall at chosen threshold"],
        deliverables=["solution.py", "fraud_report.md"],
        research_extension="Try IsolationForest as anomaly detection baseline.",
    ),
    ProjectSpec(
        id="p07-sentiment-analysis",
        title="Sentiment Analysis",
        stage="intermediate", phase=2,
        brief="Text classification on IMDB reviews (download) or a small bundled review set. TF-IDF + linear model baseline, then embeddings.",
        objectives=["Text preprocessing pipeline", "TF-IDF + LogisticRegression baseline",
                    "Compare with a neural approach"],
        dataset="imdb",
        task_breakdown=["Load/prepare text", "TF-IDF vectorization", "LogReg baseline",
                        "Optional: small PyTorch classifier", "Error analysis"],
        starter_code='''from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# TODO: sentiment pipeline
''',
        acceptance_criteria=["TF-IDF baseline > 80% acc on subset", "Vectorizer + model in pipeline",
                             "Error analysis on misclassified reviews"],
        metrics=["accuracy", "F1", "confusion matrix"],
        deliverables=["solution.py", "sentiment_report.md"],
        research_extension="Use pretrained embeddings (e.g., sentence-transformers if available).",
    ),
    ProjectSpec(
        id="p08-random-forest",
        title="Random Forest Classification",
        stage="intermediate", phase=2,
        brief="Deep dive into Random Forests: bootstrap sampling, feature importance, OOB score, hyperparameters.",
        objectives=["Understand RF mechanics", "Tune n_estimators/max_depth",
                    "Interpret feature importances (and their limits)"],
        dataset="breast_cancer",
        task_breakdown=["EDA", "RF baseline", "OOB vs CV comparison", "Hyperparameter tuning",
                        "Permutation importance vs impurity importance"],
        starter_code='''from sklearn.ensemble import RandomForestClassifier
# TODO: RF experiment on breast_cancer
''',
        acceptance_criteria=["OOB score computed", "Permutation importance computed",
                             "Hyperparameter comparison table"],
        metrics=["accuracy", "OOB score", "AUC"],
        deliverables=["solution.py", "rf_analysis.md"],
        research_extension="Compare ExtraTrees and GradientBoosting.",
    ),
    ProjectSpec(
        id="p09-time-series",
        title="Time Series Forecasting",
        stage="intermediate", phase=2,
        brief="Forecast synthetic (or real) time series: baseline, ARIMA-style decomposition, ML approach, evaluation with backtesting.",
        objectives=["Respect temporal ordering (no shuffle)", "Decompose trend/seasonality",
                    "Backtest evaluation"],
        dataset="synthetic_timeseries",
        task_breakdown=["Load series", "Decomposition plot", "Naive + seasonal-naive baselines",
                        "ML forecaster (lag features + RF/GBM)", "Walk-forward backtest"],
        starter_code='''import pandas as pd
# TODO: time series forecasting pipeline
''',
        acceptance_criteria=["Walk-forward validation used", "Naive baseline beaten",
                             "Forecast plot with confidence interval"],
        metrics=["RMSE", "MAE", "MAPE"],
        deliverables=["solution.py", "forecast_report.md", "forecast_plot.png"],
        research_extension="Add exogenous features or try Prophet if installed.",
    ),
    ProjectSpec(
        id="p10-nn-classification",
        title="Neural Network Classification (PyTorch)",
        stage="intermediate", phase=3,
        brief="Train a feedforward network on digits or Fashion-MNIST subset. Learn training loops, loss, optimizers, overfitting.",
        objectives=["Write a training loop in PyTorch", "Compare optimizers",
                    "Diagnose over/underfitting"],
        dataset="digits",
        task_breakdown=["Flatten images", "Define MLP", "Train loop with Adam",
                        "Validation curves", "Compare SGD vs Adam vs different LR"],
        starter_code='''import torch
import torch.nn as nn
# TODO: MLP classifier on digits
''',
        acceptance_criteria=[">95% test accuracy on digits", "Training/val curves plotted",
                             "At least 2 optimizer/LR configs compared"],
        metrics=["accuracy", "loss curves"],
        deliverables=["train.py", "results.md"],
        research_extension="Add dropout/weight decay; try a small CNN.",
    ),
    ProjectSpec(
        id="p11-ga-optimization",
        title="Genetic Algorithm Optimization",
        stage="advanced", phase=4,
        brief="Implement a GA from scratch (no DEAP required) to optimize a nonconvex function and a small hyperparameter search.",
        objectives=["Implement selection/crossover/mutation", "Tune GA hyperparameters",
                    "Apply to hyperparameter optimization"],
        dataset="synthetic_regression",
        task_breakdown=["Implement GA core", "Optimize sphere/Rastrigin function",
                        "Convergence analysis", "Apply to model selection"],
        starter_code='''import numpy as np
# TODO: genetic algorithm implementation
''',
        acceptance_criteria=["GA converges on Rastrigin within tolerance", "Convergence plot",
                             "Applied to at least one real optimization"],
        metrics=["best fitness over generations", "success rate"],
        deliverables=["ga.py", "ga_report.md"],
        research_extension="Add elitism, tournament vs roulette selection comparison.",
    ),
    ProjectSpec(
        id="p11b-automl",
        title="Automated ML Pipeline",
        stage="advanced", phase=4,
        brief="Build an AutoML-style pipeline: preprocessing search + model selection + HPO on a tabular dataset.",
        objectives=["Compose a search space", "Use cross-validation in loop",
                    "Log and compare trials"],
        dataset="breast_cancer",
        task_breakdown=["Define search space", "Random/grid search over pipelines",
                        "Log trials to SQLite (academy experiment log)", "Report best config"],
        starter_code='''from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
# TODO: AutoML-style search
''',
        acceptance_criteria=[">=20 configurations evaluated", "Best config beats baseline",
                             "Trials logged via academy experiment log"],
        metrics=["CV accuracy", "search efficiency"],
        deliverables=["automl.py", "search_report.md"],
        research_extension="Add successive halving or median pruning.",
    ),
    ProjectSpec(
        id="p12-genetic-programming",
        title="Genetic Programming",
        stage="advanced", phase=4,
        brief="Evolve symbolic expressions (trees) to fit data — implement or use gplearn-style GP for symbolic regression.",
        objectives=["Represent expressions as trees", "Implement subtree crossover/mutation",
                    "Fit a known function from noisy data"],
        dataset="synthetic_gp",
        task_breakdown=["Expression tree representation", "Fitness = error on data",
                        "GP loop (init, select, crossover, mutate)", "Simplification of best tree",
                        "Compare evolved vs true expression"],
        starter_code='''import numpy as np
# TODO: genetic programming for symbolic regression
''',
        acceptance_criteria=["Evolved expression approximates y within RMSE threshold",
                             "Tree visualization or readable expression printed",
                             "Random seed fixed and reported"],
        metrics=["RMSE", "tree size", "generations"],
        deliverables=["gp.py", "gp_report.md"],
        research_extension="Add parsimony pressure; test on a second ground-truth function.",
    ),
    ProjectSpec(
        id="p13-symbolic-regression",
        title="Symbolic Regression (Benchmark)",
        stage="advanced", phase=4,
        brief="Benchmark symbolic regression approaches: your GP vs sklearn alternatives (e.g., polynomial features + sparsity) on the synthetic_gp dataset and Feynman-style equations.",
        objectives=["Formalize the SR task", "Compare methods fairly",
                    "Analyze failure modes"],
        dataset="synthetic_gp",
        task_breakdown=["Implement/compare 2 SR methods", "Same train/test protocol",
                        "Complexity vs accuracy tradeoff", "Write benchmark table"],
        starter_code='''# TODO: symbolic regression benchmark
''',
        acceptance_criteria=[">=2 methods compared", "Held-out test RMSE for each",
                             "Complexity measure reported (e.g., node count)"],
        metrics=["test RMSE", "expression complexity"],
        deliverables=["sr_benchmark.py", "benchmark.md"],
        research_extension="Try SR on a real physics dataset (e.g., pendulum period vs length).",
    ),
    ProjectSpec(
        id="p14-automated-discovery",
        title="Automated Scientific Discovery",
        stage="advanced", phase=4,
        brief="End-to-end mini research pipeline: hypothesis -> data -> model -> validation -> report, automating parts with search.",
        objectives=["Frame a falsifiable hypothesis", "Automate experiment grid",
                    "Produce a structured lab report"],
        dataset="nasa_exoplanet_archive",
        task_breakdown=["Pick a research question (e.g., predict planet radius)",
                        "Download + clean NASA data", "Baseline model", "Search over features/models",
                        "Write report with claims + evidence"],
        starter_code='''# TODO: automated discovery pipeline on NASA exoplanet data
''',
        acceptance_criteria=["Hypothesis stated before experiments", "Baseline vs improved model compared",
                             "Report includes limitations section"],
        metrics=["chosen by research question"],
        deliverables=["discovery.py", "lab_report.md"],
        research_extension="Add Bayesian optimization for HPO instead of grid search.",
    ),
    ProjectSpec(
        id="p15-research-project",
        title="Original Research Project",
        stage="advanced", phase=6,
        brief="Full research cycle: problem -> hypothesis -> experiment -> ablation -> paper draft.",
        objectives=["Execute the Phase 5-6 methodology", "Reproducible experiment",
                    "Paper draft following a template"],
        dataset="kepler_koi",
        task_breakdown=["Define research question", "Literature mini-review",
                        "Experimental design + pre-registration (write hypothesis first)",
                        "Run experiments + ablations", "Draft paper (IMRaD)"],
        starter_code='''# TODO: original research project — see academy/projects/research_template.md
''',
        acceptance_criteria=["Hypothesis written before results", "Ablation study included",
                             "Reproducible: seed + requirements + commands documented",
                             "Paper draft follows IMRaD"],
        metrics=["chosen by research question"],
        deliverables=["research/ directory: proposal.md, experiments/, paper.md"],
        research_extension="Submit-ready polish: figures, related work, threats to validity.",
    ),
]


def get_project(pid: str) -> ProjectSpec:
    for p in PROJECTS:
        if p.id == pid:
            return p
    raise KeyError(f"Project not found: {pid}")


def list_projects() -> list[ProjectSpec]:
    return PROJECTS
