# Machine-Learning-Project
Machine learning project focused on using MMA stats to predict fight outcomes.

## Project Structure
```text
MACHINE-LEARNING-PROJECT/
├── data/
│   ├── raw/
│   │   ├── ufc_event_fight_stats.csv
│   │   ├── ufc_events.csv
│   │   ├── ufc_fighters.csv
│   │   ├── ufc_fighters_avg.csv
│   │   ├── ufc_fighters_median.csv
│   │   └── ufc_fights_with_betting_odds.csv
│   ├── processed/
│   │   ├── fights_cleaned.csv
│   │   ├── fights_testing.csv
│   │   ├── fights_testing_exp1.csv
│   │   ├── fights_testing_exp2.csv
│   │   ├── fights_training.csv
│   │   ├── fights_training_exp1.csv
│   │   ├── fights_training_exp2.csv
│   │   ├── fights_validation.csv
│   │   ├── fights_validation_exp1.csv
│   │   ├── fights_validation_exp2.csv
│   │   ├── odds_cleaned.csv
│   │   ├── random_forest_validation_predictions.csv
│   │   ├── Logistic_Regression_Outputs/
│   │   │   ├── logistic_regression_test_predictions.csv
│   │   │   ├── logistic_regression_test_exp1_predictions.csv
│   │   │   ├── logistic_regression_test_exp2_predictions.csv
│   │   │   ├── logistic_regression_validation_predictions.csv
│   │   │   ├── logistic_regression_validation_exp1_predictions.csv
│   │   │   └── logistic_regression_validation_exp2_predictions.csv
│   │   └── XGBoost_Outputs/
│   │       ├── xgboost_test_predictions.csv
│   │       ├── xgboost_test_exp1_predictions.csv
│   │       ├── xgboost_test_exp2_predictions.csv
│   │       ├── xgboost_validation_predictions.csv
│   │       ├── xgboost_validation_exp1_predictions.csv
│   │       └── xgboost_validation_exp2_predictions.csv
│   └── scaler/
│       ├── scaler.pkl
│       ├── scaler_exp1.pkl
│       └── scaler_exp2.pkl
├── notebooks/
│   ├── data_cleaning.ipynb
│   ├── feature_engineering.ipynb
│   ├── logistic_regression.ipynb
│   ├── model_random_forest.ipynb
│   ├── model_xgboost.ipynb
│   ├── odds_data_pre.ipynb
│   └── playAround.py.text
├── models/
│   ├── logistic_regression_baseline.joblib
│   └── random_forest.joblib
├── .gitignore
├── requirements.txt
└── README.md
```

Ignored files and folders such as `.venv/`, `.ipynb_checkpoints/`, `__pycache__/`, `*.pyc`, `.env`, `kaggle.json`, `predict_upcoming_fights.py`, and `scripts/` are intentionally omitted from this structure.

## Setup
```bash
pip install -r requirements.txt
```

## Notebook Order
1. `data_cleaning.ipynb` — clean raw data, outputs to `data/processed/`
2. `feature_engineering.ipynb` — build features, outputs to `data/processed/`
3. `odds_data_pre.ipynb` — prepare betting odds data for downstream analysis
4. `logistic_regression.ipynb` — train and evaluate the baseline logistic regression model
5. `model_random_forest.ipynb` — train and evaluate the random forest model
6. `model_xgboost.ipynb` — train and evaluate the XGBoost model
