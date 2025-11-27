import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
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


def main():
    train_path = "train.csv"
    test_path = "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_raw = train_df["Status"]
    X_train_df = train_df.drop(columns=["Status", "id"])
    X_test_df = test_df.drop(columns=["id"])

    categorical_cols = X_train_df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [c for c in X_train_df.columns if c not in categorical_cols]

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    preprocessor.fit(X_train_df)

    X_all = preprocessor.transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(y_raw)

    # Holdout validation for early stopping and to report log loss
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=77
    )

    base_params = dict(
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=0.8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=3,
        tree_method="hist",
        reg_lambda=1.0,
        reg_alpha=0.1,
        gamma=0.05,
        early_stopping_rounds=80,
        n_jobs=4,
    )

    val_model = XGBClassifier(**base_params)
    val_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    val_preds = val_model.predict_proba(X_val)
    val_loss = log_loss(y_val, val_preds)
    best_round = (
        val_model.best_iteration + 1 if val_model.best_iteration is not None else base_params["n_estimators"]
    )
    print(f"Holdout log loss: {val_loss:.4f} with {best_round} boosting rounds")

    # Refit on all data with the tuned number of trees
    final_params = dict(base_params)
    final_params.pop("early_stopping_rounds", None)
    final_params["n_estimators"] = best_round
    final_model = XGBClassifier(**final_params)
    final_model.fit(X_all, y_all)

    test_proba = final_model.predict_proba(X_test)
    class_order = ["C", "CL", "D"]
    proba_df = pd.DataFrame(test_proba, columns=label_encoder.classes_)
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
    print("Saved submission.csv")


if __name__ == "__main__":
    main()
