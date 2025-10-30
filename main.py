# RADIOMICS PIPELINE — Load Data + Split + Preprocess + FS + Modeling + Halving + Explainability

import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score

# === Import internal modules ===
from src.load_data import load_and_clean
from src.split_and_check import split_and_check
from src.preprocessing import variance_filter, correlation_filter, stat_filter, PowerTransformer
from src.feature_selection import (
    fs_mrmr, fs_relieff, fs_corrsf, fs_ses, fs_boruta,
    fs_rfe_svm, fs_genetic, fs_lasso, fs_hsic_lasso, fs_rf_importance
)
from src.models import get_models_and_params
from src.evaluation import run_experiments  
from src import explainability            

# === Paths === #
base = Path("data")
path = base / "radiomics features.xlsx"
split_dir = base / "split_report"
features_dir = base / "selected_features"
results_dir = base / "model_results"
for d in [split_dir, features_dir, results_dir]:
    d.mkdir(parents=True, exist_ok=True)

# === Load dataset === #
print("Checking data path:")
if not path.exists():
    raise FileNotFoundError(f"File not found: {path.resolve()}")

print(f"Found dataset: {path.name}")
print("\n Loading and Cleaning Dataset: ")
X, y = load_and_clean(path)

# === Load and align center info === #
print("\n Loading center information: ")
try:
    df_centers = pd.read_excel(path, usecols=["center"])
    centers = df_centers.loc[X.index, "center"].astype(str)
    print(f"   Loaded {centers.nunique()} unique centers: {centers.unique().tolist()}")
    print(f"   Centers aligned with dataset: {len(centers)} entries.")
except Exception as e:
    print(f" No 'center' column found or could not load centers: {e}")
    centers = None

# === Split & Heterogeneity Check === #
print("\n Creating stratified & grouped folds by center: ")
splits, fold_assignments, report = split_and_check(
    X, y, centers=centers, n_splits=3, random_state=42, output_dir=split_dir
)

print("\n Heterogeneity Summary:")
print(f"   mean_label_std:  {report['mean_label_std']:.2f}%")
if "mean_center_std" in report and not pd.isna(report["mean_center_std"]):
    print(f"   mean_center_std: {report['mean_center_std']:.2f}%")
print(f"\n Split & Heterogeneity check completed.")
print(f"   Heatmaps saved in: {split_dir.resolve()}")

# === Preprocessing === #
print("\n Starting preprocessing: ")
pt = PowerTransformer(method='yeo-johnson', standardize=True)
X = pd.DataFrame(pt.fit_transform(X), columns=X.columns, index=X.index)

print(" VarianceThreshold filter...")
X = variance_filter(X, threshold=0.01)
print(f"   Remaining features: {X.shape[1]}")

print(" Removing highly correlated features: ")
X = correlation_filter(X, threshold=0.85)
print(f"   Remaining features: {X.shape[1]}")

print(" Kruskal/Mann–Whitney filtering: ")
X = stat_filter(X, y, alpha=0.1)
print(f"   Remaining features: {X.shape[1]}")

print("\n Preprocessing completed successfully.")
print(f"   Final feature count: {X.shape[1]}")

# === Feature Selection === #
print("\n Running Feature Selection methods...")

methods = {
    "mRMR": lambda: fs_mrmr(X, y, top_k=20),
    "ReliefF": lambda: fs_relieff(X, y, top_k=20),
    "CorrSF": lambda: fs_corrsf(X, y, top_k=20),
    "SES": lambda: fs_ses(X, y, alpha=0.1),
    "Boruta": lambda: fs_boruta(X, y),
    "RFE-SVM": lambda: fs_rfe_svm(X, y, n_features=20),
    "Genetic": lambda: fs_genetic(X, y, top_k=20),
    "LASSO": lambda: fs_lasso(X, y),
    "HSIC-LASSO": lambda: fs_hsic_lasso(X, y, top_k=20),
    "RF-Importance": lambda: fs_rf_importance(X, y, top_k=20)
}

selected_datasets = {}
for name, func in methods.items():
    try:
        selected = func()
        selected_datasets[name] = X[selected]
        pd.Series(selected).to_csv(features_dir / f"selected_{name}.csv", index=False)
        print(f"  {name}: {len(selected)} features saved.")
    except Exception as e:
        print(f"    {name} failed: {e}")

print("\n  Feature selection completed.")
print(f"   Results saved to: {features_dir.resolve()}")

# === MODEL TRAINING (CROSS-VALIDATION) === #
print("\n Starting model evaluation across feature selection methods: ")

models, params = get_models_and_params()
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scorer = make_scorer(f1_score, average="weighted")

results = []
for fs_name, X_fs in selected_datasets.items():
    print(f"\n Evaluating models using features from {fs_name} ({X_fs.shape[1]} features)...")
    for model_name, model in models.items():
        try:
            pipeline = Pipeline([("clf", model)])
            scores = cross_val_score(pipeline, X_fs, y, cv=cv, scoring=scorer, n_jobs=-1)
            acc_scores = cross_val_score(pipeline, X_fs, y, cv=cv, scoring="accuracy", n_jobs=-1)
            results.append({
                "FeatureSelection": fs_name,
                "Model": model_name,
                "F1_mean": scores.mean(),
                "F1_std": scores.std(),
                "Accuracy_mean": acc_scores.mean()
            })
            print(f"    {model_name}: F1={scores.mean():.3f} ± {scores.std():.3f} | Acc={acc_scores.mean():.3f}")
        except Exception as e:
            print(f"    {model_name} failed: {e}")

df_results = pd.DataFrame(results)
df_results.to_csv(results_dir / "model_comparison.csv", index=False)

print("\n Model comparison summary:")
print(df_results.sort_values(by="F1_mean", ascending=False).head(10))
print(f"\n All results saved in: {results_dir.resolve()}")

# ===  ADVANCED EVALUATION (HALVING RANDOM SEARCH) === #
print("\n Starting advanced evaluation with Halving Random Search: ")

# Επιλογή top Feature Selection sets
top_fs = {k: v for k, v in selected_datasets.items() if k in ["LASSO", "RFE-SVM", "SES"]}

# Advanced search
halving_results = run_experiments(
    selected_datasets=top_fs,
    y=y,
    models=models,
    param_grids=params,
    cv=3
)

print("\n Halving Search completed successfully!")
print(halving_results.sort_values(by='F1_score', ascending=False).head(10))

# ===  EXPLAINABILITY (SHAP + LIME) === #
try:
    print("\n Launching explainability analysis (SHAP + LIME)...")
    explainability.run_explainability()
    print("\n Explainability module completed successfully!")
except Exception as e:
    print(f" Explainability analysis skipped due to error: {e}")

print("\n  Full radiomics pipeline completed successfully! ")

