import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import os

def load_and_clean(path, id_col="case_id", target_col="label", min_class_size=10):
    
    # --- Load Excel dataset ---
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except ImportError:
        raise ImportError("Please install 'openpyxl': pip install openpyxl")
    except Exception as e:
        raise RuntimeError(f"Error reading Excel file: {e}")

    print(f"\n Loaded file: {os.path.basename(path)} | Shape: {df.shape}")

    # --- Separate features and labels ---
    y = df[target_col].copy()
    X = df.drop(columns=[c for c in [id_col, target_col] if c in df.columns])
    X = X.select_dtypes(include=[np.number])  # keep numeric only

    # --- Class distribution before filtering ---
    unique, counts = np.unique(y, return_counts=True)
    print("\n Class distribution before filtering:")
    for cls, count in zip(unique, counts):
        print(f"   {cls}: {count} samples")

    # --- Filter small classes ---
    small_classes = [cls for cls, c in zip(unique, counts) if c < min_class_size]
    if small_classes:
        print(f"\n Removing small classes (<{min_class_size} samples): {small_classes}")
        mask = ~y.isin(small_classes)
        X, y = X[mask], y[mask]
        print(f"   New shape: {X.shape}, remaining classes: {sorted(y.unique())}")
    else:
        print(" All classes have sufficient samples.")

    # --- Encode labels ---
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(f" Encoded classes: {list(le.classes_)} → {list(range(len(le.classes_)))}")

    # --- Replace Inf and drop empty columns ---
    X = X.replace([np.inf, -np.inf], np.nan)
    dropped = X.columns[X.isna().all()].tolist()
    if dropped:
        print(f"\n  Dropped {len(dropped)} empty columns: {dropped[:5]}{'...' if len(dropped) > 5 else ''}")
        X = X.dropna(axis=1, how="all")

    # --- Impute missing values ---
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    print(f"\n Clean dataset ready: {X.shape[0]} samples × {X.shape[1]} features")
    return X, y
