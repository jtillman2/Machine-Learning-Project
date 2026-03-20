# Machine-Learning-Project
Machine learning project focused on using MMA stats to predict fight outcomes.

## Project Structure
```text
MACHINE-LEARNING-PROJECT/
├── data/
│   ├── raw/                        <- original CSVs, do not modify
│   │   ├── ufc_event_fight_stats.csv
│   │   ├── ufc_events.csv
│   │   ├── ufc_fighters.csv
│   │   ├── ufc_fighters_avg.csv
│   │   └── ufc_fighters_median.csv
│   ├── processed/                  <- cleaned/feature-engineered CSVs saved here
│   │   ├── fights_cleaned.csv
│   │   ├── fights_training.csv
│   │   ├── fights_validation.csv
│   │   └── fights_testing.csv
│   └── scaler/                     <- fitted scaler artifacts
├── notebooks/
│   ├── data_cleaning.ipynb
│   ├── feature_engineering.ipynb
│   └── logistic_regression.ipynb
├── models/
│   └── logistic_regression_baseline.joblib
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
```

## Notebook Order
1. `data_cleaning.ipynb` — clean raw data, outputs to `data/processed/`
2. `feature_engineering.ipynb` — build features, outputs to `data/processed/`
3. `logistic_regression.ipynb` — train and evaluate baseline logistic regression
