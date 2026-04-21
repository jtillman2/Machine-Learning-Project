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

## Predict Upcoming Fights (Script)
Use the script below to predict future fight winners from fighter A/B stats in a CSV.

1. Fill in `data/processed/upcoming_fights_template.csv` (one row per fight).
2. Run:

```bash
.venv/bin/python predict_upcoming_fights.py \
	--input data/processed/upcoming_fights_template.csv \
	--output data/processed/upcoming_fights_predictions.csv
```

Output includes:
- `predicted_winner_name`
- `prob_fighter_a_win`
- `confidence`

Notes:
- If `weight_class` is provided, class encodings are inferred from historical data.
- Stances are normalized to: `Orthodox`, `Southpaw`, `Switch`, or `Other`.
