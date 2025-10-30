
# Import libraries + packages
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, StackingClassifier,
    GradientBoostingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def get_models_and_params():

    # -------------------- EXISTING MODELS -------------------- #
    # Random Forest
    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    )

    # Optimized SVM (RBF)
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.9, random_state=42)),
        ("clf", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # Stacking Ensemble (RF + SVM + Gradient Boosting)
    stacking_model = StackingClassifier(
        estimators=[
            ("rf", rf),
            ("svm", svm)
        ],
        final_estimator=GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),
        passthrough=True,
        n_jobs=-1
    )

    # Soft Voting Ensemble
    soft_voting = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("svm", svm)
        ],
        voting="soft",
        weights=[1, 1],
        n_jobs=-1
    )

    # -------------------- NEW MODELS -------------------- #

    # Logistic Regression (L2)
    log_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty='l2',
            solver='liblinear',
            class_weight='balanced',
            random_state=42
        ))
    ])

    # XGBoost
    xgb = XGBClassifier(
        clf__n_estimators=500,
        clf__learning_rate=0.05,
        clf__max_depth=5,
        clf__subsample=0.8,
        clf__colsample_bytree=0.8,
        clf__eval_metric='mlogloss',
        clf__random_state=42
    )

    # LightGBM
    lgbm = LGBMClassifier(
        clf__n_estimators=500,
        clf__learning_rate=0.05,
        clf__max_depth=-1,
        clf__class_weight='balanced',
        clf__random_state=42
    )

    # k-Nearest Neighbors
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5, weights='distance'))
    ])

    # Multi-layer Perceptron (Neural Network)
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        ))
    ])

    # -------------------- MODELS DICTIONARY -------------------- #
    models = {
        "Random Forest": rf,
        "SVM (RBF)": svm,
        "Stacking Ensemble (RF+SVM+GB)": stacking_model,
        "Soft Voting (RF+SVM)": soft_voting,
        "Logistic Regression": log_reg,
        "XGBoost": xgb,
        "LightGBM": lgbm,
        "kNN": knn,
        "MLP (Neural Net)": mlp
    }

    # -------------------- HYPERPARAMETER GRIDS -------------------- #
    params = {
        "Random Forest": {
            "clf__n_estimators": [300, 600, 1000],
            "clf__max_depth": [10, 20, None],
            "clf__min_samples_split": [2, 5],
            "clf__min_samples_leaf": [1, 2],
            "clf__max_features": ["sqrt", "log2"],
        },

        "SVM (RBF)": {
            "clf__clf__C": [0.1, 1, 10, 50, 100],
            "clf__clf__gamma": [1e-4, 1e-3, 0.01, 0.1, "scale"],
            "clf__pca__n_components": [0.85, 0.9, 0.95],
        },

        

        "Stacking Ensemble (RF+SVM+GB)": {
            "clf__final_estimator__n_estimators": [100, 200, 300],
            "clf__final_estimator__learning_rate": [0.03, 0.05, 0.1],
            "clf__final_estimator__max_depth": [2, 3, 4],
        },

        "Soft Voting (RF+SVM)": {
            "clf__weights": [(1, 1), (2, 1), (1, 2)]
        },

        "Logistic Regression": {
            "clf__clf__C": [0.01, 0.1, 1, 10, 100]
        },

        "XGBoost": {
            "clf__n_estimators": [300, 500, 800],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__max_depth": [3, 5, 7],
            "clf__subsample": [0.7, 0.8, 1.0],
            "clf__colsample_bytree": [0.7, 0.8, 1.0]
        },

        "LightGBM": {
            "clf__n_estimators": [300, 500, 800],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__max_depth": [-1, 5, 10],
            "clf__num_leaves": [31, 63, 127]
        },

        "kNN": {
            "clf__clf__n_neighbors": [3, 5, 7, 9],
            "clf__clf__weights": ["uniform", "distance"]
        },

        "MLP (Neural Net)": {
            "clf__clf__hidden_layer_sizes": [(64,), (128, 64), (128, 64, 32)],
            "clf__clf__activation": ["relu", "tanh"],
            "clf__clf__learning_rate_init": [0.001, 0.01]
        }
    }

    return models, params


if __name__ == "__main__":
    models, params = get_models_and_params()
    print("✅ Available models:")
    for name in models.keys():
        print(f" - {name}")
    print("\nParameters for SVM:")
    print(params["SVM (RBF)"])
