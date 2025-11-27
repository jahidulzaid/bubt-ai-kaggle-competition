import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def build_preprocessor(cat_cols, num_cols):
    """Create preprocessing pipeline for numeric and categorical fields."""
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def train_cv_ensemble(train_df, test_df, params, n_splits=5, random_state=42):
    """Cross-validated training with fold-level models averaged for robustness."""
    y_raw = train_df["Status"]
    X = train_df.drop(columns=["Status", "id"])
    X_test_raw = test_df.drop(columns=["id"])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof = np.zeros((len(train_df), len(label_encoder.classes_)))
    test_pred = np.zeros((len(test_df), len(label_encoder.classes_)))
    fold_best_rounds = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        preprocessor = build_preprocessor(cat_cols, num_cols)
        X_tr_proc = preprocessor.fit_transform(X_tr)
        X_va_proc = preprocessor.transform(X_va)
        X_test_proc = preprocessor.transform(X_test_raw)

        model = XGBClassifier(**params)
        model.fit(X_tr_proc, y_tr, eval_set=[(X_va_proc, y_va)], verbose=False)

        va_pred = model.predict_proba(X_va_proc)
        oof[va_idx] = va_pred
        fold_loss = log_loss(y_va, va_pred, labels=[0, 1, 2])
        best_round = (
            model.best_iteration + 1
            if getattr(model, "best_iteration", None) is not None
            else params["n_estimators"]
        )
        fold_best_rounds.append(best_round)
        print(f"Fold {fold}: logloss={fold_loss:.4f}, rounds={best_round}")

        test_pred += model.predict_proba(X_test_proc)

    test_pred /= n_splits
    cv_loss = log_loss(y, oof, labels=[0, 1, 2])
    avg_rounds = int(np.round(np.mean(fold_best_rounds)))
    print(f"CV logloss={cv_loss:.4f}, avg best rounds={avg_rounds}")

    return oof, test_pred, label_encoder, avg_rounds, cv_loss


def main():
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    base_params = dict(
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=0.8,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=3,
        tree_method="hist",
        reg_lambda=1.2,
        reg_alpha=0.15,
        gamma=0.05,
        early_stopping_rounds=80,
        n_jobs=4,
    )

    oof, test_pred, label_encoder, avg_rounds, cv_loss = train_cv_ensemble(
        train_df, test_df, base_params, n_splits=5, random_state=99
    )

    # Optional final fit on full data with averaged best rounds
    cat_cols = train_df.drop(columns=["Status", "id"]).select_dtypes(include=["object"]).columns.tolist()
    num_cols = [
        c for c in train_df.drop(columns=["Status", "id"]).columns if c not in cat_cols
    ]
    preprocessor = build_preprocessor(cat_cols, num_cols)
    X_all = preprocessor.fit_transform(train_df.drop(columns=["Status", "id"]))
    X_test = preprocessor.transform(test_df.drop(columns=["id"]))

    final_params = dict(base_params)
    final_params.pop("early_stopping_rounds", None)
    final_params["n_estimators"] = max(avg_rounds, 50)
    final_model = XGBClassifier(**final_params)
    y_all = label_encoder.transform(train_df["Status"])
    final_model.fit(X_all, y_all)
    final_test_pred = final_model.predict_proba(X_test)

    blended_test_pred = (test_pred + final_test_pred) / 2

    class_order = ["C", "CL", "D"]
    proba_df = pd.DataFrame(blended_test_pred, columns=label_encoder.classes_)
    proba_df = proba_df[class_order]

    submission = pd.DataFrame(
        {
            "id": test_df["id"],
            "Status_C": proba_df["C"].clip(1e-15, 1 - 1e-15),
            "Status_CL": proba_df["CL"].clip(1e-15, 1 - 1e-15),
            "Status_D": proba_df["D"].clip(1e-15, 1 - 1e-15),
        }
    )
    submission.to_csv("submission.csv", index=False)
    print(f"Saved submission.csv | CV logloss {cv_loss:.4f}")


if __name__ == "__main__":
    main()
