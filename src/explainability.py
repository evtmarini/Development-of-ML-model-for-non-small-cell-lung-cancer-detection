# ==========================================
# explainability.py
# Standalone script for model explainability (SHAP + LIME)
# ==========================================

import pandas as pd
from ast import literal_eval
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np


def run_explainability():
    print("🚀 Running standalone explainability pipeline...\n")

    warnings.filterwarnings("ignore", message="X has feature names")

    # ================================================================
    # 🧩 STEP 1: Load halving results and find the best combination
    # ================================================================
    halving_path = Path("data/halving_results.csv")
    if not halving_path.exists():
        raise FileNotFoundError(f"❌ halving_results.csv not found in {halving_path.resolve()}")

    df = pd.read_csv(halving_path)

    best_row = df.loc[df["F1_score"].idxmax()]
    best_fs = best_row["FS_method"]
    best_model = best_row["Classifier"]
    best_f1 = best_row["F1_score"]
    best_params = literal_eval(str(best_row["Best_params"]))

    print("🏆 Best pipeline found:")
    print(f"   ➤ Feature Selection: {best_fs}")
    print(f"   ➤ Classifier: {best_model}")
    print(f"   ➤ F1-score: {best_f1:.4f}")
    print(f"   ➤ Best Params: {best_params}")

    # ================================================================
    # 🧩 STEP 2: Load selected feature set and target labels
    # ================================================================
    features_dir = Path("data/selected_features")
    fs_file = features_dir / f"selected_{best_fs}.csv"

    if not fs_file.exists():
        raise FileNotFoundError(f"❌ Selected features file not found: {fs_file.resolve()}")

    print(f"\n📂 Loading feature file: {fs_file.name}")
    selected_data = pd.read_csv(fs_file)

    data_path = Path("data/radiomics features.xlsx")
    if not data_path.exists():
        raise FileNotFoundError(f"❌ Radiomics dataset not found at {data_path.resolve()}")

    if selected_data.shape[1] == 1:
        feature_names = selected_data.iloc[:, 0].tolist()
        full_df = pd.read_excel(data_path)
        y = full_df["label"]
        X = full_df[feature_names]
        print(f"✅ Loaded dataset from Excel using {len(feature_names)} selected features.")
    else:
        X = selected_data
        y = pd.read_excel(data_path)["label"]
        print(f"✅ Loaded full matrix directly from {fs_file.name}")

    print(f"📊 X shape: {X.shape}")
    print(f"🎯 y shape: {y.shape}")
    print("\n✅ Data loaded successfully. Ready for explainability analysis.")

    # ================================================================
    # 🧩 STEP 3: Train best model and compute SHAP
    # ================================================================
    print("\n🧠 Training best model and computing SHAP values...")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    gb = GradientBoostingClassifier(random_state=42)

    stacking_model = StackingClassifier(
        estimators=[("rf", rf), ("svm", svm)],
        final_estimator=GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42
        ),
        passthrough=True,
        n_jobs=-1
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", stacking_model)
    ])

    model.fit(X_train, y_train)
    print("✅ Model trained successfully.")

    # ================================================================
    # 🧩 STEP 4: SHAP Explainability
    # ================================================================
    os.makedirs("results_explainability", exist_ok=True)
    print("\n🔍 Running SHAP explainability...")

    stack_clf = model.named_steps["clf"]
    X_sample = X_train.sample(min(100, len(X_train)), random_state=42)
    explainer = shap.Explainer(stack_clf.predict_proba, X_sample)

    X_test_sample = X_test.sample(min(50, len(X_test)), random_state=42)
    shap_values = explainer(X_test_sample)

    class_names = list(label_encoder.classes_)
    print(f"   ➤ Classes: {class_names}")
    print(f"   ➤ SHAP shape: {np.array(shap_values.values).shape}")

    # Summary plot
    shap.summary_plot(
        shap_values.values if shap_values.values.ndim == 2 else shap_values[..., 0],
        X_test_sample,
        show=False,
        plot_size=(10, 6)
    )
    plt.tight_layout()
    plt.savefig("results_explainability/shap_summary_plot.png", dpi=300)
    plt.close()

    # Bar plot
    shap.summary_plot(
        shap_values.values if shap_values.values.ndim == 2 else shap_values[..., 0],
        X_test_sample,
        show=False,
        plot_type="bar",
        plot_size=(10, 6)
    )
    plt.tight_layout()
    plt.savefig("results_explainability/shap_bar_plot.png", dpi=300)
    plt.close()

    print("📊 Saved SHAP summary and bar plots.")

    # ================================================================
    # 🧩 STEP 5: LIME (optional)
    # ================================================================
    try:
        from lime.lime_tabular import LimeTabularExplainer
        print("\n🧠 Running LIME local explanation...")
        lime_explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=X.columns,
            class_names=class_names,
            mode="classification"
        )
        exp = lime_explainer.explain_instance(X_test_sample.iloc[0].values, stack_clf.predict_proba)
        os.makedirs("results_explainability/extended", exist_ok=True)
        exp.save_to_file("results_explainability/extended/lime_example.html")
        print("✅ LIME explanation saved as HTML.")
    except Exception as e:
        print(f"⚠️ LIME skipped: {e}")

    print("\n🏁 Explainability module completed successfully.")


# ================================================================
# Entry point (so it can run standalone)
# ================================================================
if __name__ == "__main__":
    run_explainability()
