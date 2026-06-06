"""
Train and save the XGBoost Exp1 model.

Matches the hyperparameters and data splits from notebooks/model_xgboost.ipynb.
Saves the trained model to models/xgboost_exp1.joblib.

Run from project root:
  python -m models.train_xgboost
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"

TRAIN_PATH = DATA / "fights_training_exp1.csv"
VAL_PATH   = DATA / "fights_validation_exp1.csv"
TEST_PATH  = DATA / "fights_testing_exp1.csv"
OUT_PATH   = MODELS / "xgboost_exp1.joblib"

DROP_COLS = ["winner", "is_mirror", "fight_id"]


def main():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    val   = pd.read_csv(VAL_PATH)
    test  = pd.read_csv(TEST_PATH)

    X_train = train.drop(columns=DROP_COLS)
    y_train = train["winner"]
    X_val   = val.drop(columns=DROP_COLS)
    y_val   = val["winner"]
    X_test  = test.drop(columns=DROP_COLS)
    y_test  = test["winner"]

    print(f"Train: {X_train.shape[0]} rows | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.649,
        colsample_bytree=0.649,
        random_state=42,
        early_stopping_rounds=20,
        eval_metric="logloss",
    )

    print("Training...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=25,
    )

    val_acc  = accuracy_score(y_val,  model.predict(X_val))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\nVal accuracy:  {val_acc:.2%}")
    print(f"Test accuracy: {test_acc:.2%}")
    print(f"Best iteration: {model.best_iteration}")

    joblib.dump(model, OUT_PATH)
    # Save feature names separately so predict.py can validate column order
    joblib.dump(list(X_train.columns), MODELS / "xgboost_exp1_features.joblib")
    print(f"\nModel saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
