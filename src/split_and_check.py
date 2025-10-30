import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import os

def split_and_check(X, y, centers=None, n_splits=3, random_state=42, n_trials=20, output_dir="data/split_report"):

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n Creating stratified folds ({n_splits}-fold CV)...")

    y = np.array(y)
    best_std, best_seed, best_splits, best_folds, best_report = np.inf, None, None, None, None

    # Multiple seeds to find the most balanced split
    for trial in range(n_trials):
        seed = random_state + trial
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_assignments = np.zeros(len(y), dtype=int)
        splits = []

        for fold_idx, (_, val_idx) in enumerate(cv.split(X, y)):
            fold_assignments[val_idx] = fold_idx + 1
            splits.append(val_idx)

        df_summary = pd.DataFrame({"Fold": fold_assignments, "Label": y})
        label_dist = pd.crosstab(df_summary["Fold"], df_summary["Label"], normalize="index") * 100
        label_std = label_dist.std(axis=0).mean()

        if label_std < best_std:
            best_std, best_seed, best_splits, best_folds = label_std, seed, splits, fold_assignments
            best_report = {"mean_label_std": label_std, "best_seed": seed}

    print(f" Best stratified split found at seed={best_seed}")
    print(f"   mean_label_std={best_report['mean_label_std']:.2f}%")

    # Verify uniqueness
    all_indices = np.concatenate(best_splits)
    if len(all_indices) == len(np.unique(all_indices)):
        print("Verified: all samples are unique across folds (no overlaps).")
    else:
        print(" Warning: overlap detected between folds!")

    # === Summary DataFrame ===
    df_summary = pd.DataFrame({"Fold": best_folds, "Label": y})
    if centers is not None:
        df_summary["Center"] = np.array(centers)
    else:
        df_summary["Center"] = "N/A"

    # Label composition
    label_counts = pd.crosstab(df_summary["Fold"], df_summary["Label"])
    print("\n Fold composition (sample counts):")
    for fold in label_counts.index:
        total = label_counts.loc[fold].sum()
        per_class = " | ".join([f"class_{c}={label_counts.loc[fold, c]}" for c in label_counts.columns])
        print(f"   Fold {fold} → total={total} | {per_class}")

    # Center composition
    if centers is not None and len(np.unique(centers)) > 1:
        print("\n Fold composition per center:")
        center_counts = pd.crosstab(df_summary["Fold"], df_summary["Center"])
        for fold in center_counts.index:
            total = center_counts.loc[fold].sum()
            per_center = " | ".join([f"{c}={center_counts.loc[fold, c]}" for c in center_counts.columns])
            print(f"   Fold {fold} → total={total} | {per_center}")

        # Center coverage check
        for fold in center_counts.index:
            missing = [c for c in center_counts.columns if center_counts.loc[fold, c] == 0]
            if missing:
                print(f" Fold {fold} is missing samples from centers: {missing}")

        # Compute heterogeneity
        center_dist = pd.crosstab(df_summary["Fold"], df_summary["Center"], normalize="index") * 100
        mean_center_std = center_dist.std(axis=0).mean()
        best_report["mean_center_std"] = mean_center_std
        print(f"\n mean_center_std: {mean_center_std:.2f}% (center distribution variability)")

        # Save heatmap
        plt.figure(figsize=(10, 4))
        sns.heatmap(center_dist, annot=True, fmt=".1f", cmap="mako")
        plt.title("Center (%) distribution per Fold — Grouped by center")
        plt.xlabel("Center")
        plt.ylabel("Fold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "heterogeneity_centers_heatmap.png"), dpi=300)
        plt.close()

    # === Label distribution heatmap ===
    label_dist = pd.crosstab(df_summary["Fold"], df_summary["Label"], normalize="index") * 100
    plt.figure(figsize=(8, 4))
    sns.heatmap(label_dist, annot=True, fmt=".1f", cmap="viridis")
    plt.title("Class (%) distribution per Fold — Stratified by label")
    plt.xlabel("Label")
    plt.ylabel("Fold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heterogeneity_labels_heatmap.png"), dpi=300)
    plt.close()
    print(f"\n Saved heatmaps to: {output_dir}")

    return best_splits, best_folds, best_report

