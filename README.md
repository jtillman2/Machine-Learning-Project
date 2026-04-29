# Machine-Learning-Project
Machine learning project focused on using MMA stats to predict UFC fight outcomes. Three model families are trained and evaluated across three dataset experiments: a base dataset (all fights), Experiment 1 (both fighters have ≥1 prior UFC fight), and Experiment 2 (both fighters have ≥2 prior UFC fights).

## Models
- **Logistic Regression** — baseline classifier
- **Random Forest** — ensemble model with hyperparameter tuning via `RandomizedSearchCV`
- **XGBoost** — gradient-boosted trees with hyperparameter tuning via Optuna

## Setup
```bash
pip install -r requirements.txt
```

## Notebook Order
1. `notebooks/data_cleaning.ipynb` — clean raw data, outputs to `data/processed/`
2. `notebooks/feature_engineering.ipynb` — engineer features and create train/val/test splits
3. `notebooks/odds_data_pre.ipynb` — clean and align betting odds with fight stats
4. `notebooks/logistic_regression.ipynb` — train and evaluate baseline logistic regression
5. `notebooks/model_random_forest.ipynb` — train and evaluate Random Forest across all experiments
6. `notebooks/model_xgboost.ipynb` — train and evaluate XGBoost across all experiments

## Directory and File Descriptions

### `/data/`
Top-level data directory containing all raw inputs and processed outputs.

#### `/data/raw/`
Original source CSVs. Do not modify these files.

`/data/raw/ufc_event_fight_stats.csv`: Per-fight statistics for each UFC event (strikes, takedowns, control time, etc.).

`/data/raw/ufc_events.csv`: Metadata for each UFC event (name, date, location, URL).

`/data/raw/ufc_fighters.csv`: Fighter profile data (name, height, reach, stance, date of birth).

`/data/raw/ufc_fighters_avg.csv`: Per-fighter average statistics computed across all historical fights.

`/data/raw/ufc_fighters_median.csv`: Per-fighter median statistics computed across all historical fights.

`/data/raw/ufc_fights_with_betting_odds.csv`: Historical American-format betting odds for UFC fights.

#### `/data/processed/`
Cleaned and feature-engineered CSVs produced by the notebooks.

`/data/processed/fights_cleaned.csv`: Cleaned and merged fight dataset output by `data_cleaning.ipynb`.

`/data/processed/fights_training.csv`: Training split for the base dataset (all fights).

`/data/processed/fights_validation.csv`: Validation split for the base dataset.

`/data/processed/fights_testing.csv`: Test split for the base dataset.

`/data/processed/fights_training_exp1.csv`: Training split for Experiment 1 (both fighters with ≥1 prior UFC fight).

`/data/processed/fights_validation_exp1.csv`: Validation split for Experiment 1.

`/data/processed/fights_testing_exp1.csv`: Test split for Experiment 1.

`/data/processed/fights_training_exp2.csv`: Training split for Experiment 2 (both fighters with ≥2 prior UFC fights).

`/data/processed/fights_validation_exp2.csv`: Validation split for Experiment 2.

`/data/processed/fights_testing_exp2.csv`: Test split for Experiment 2.

`/data/processed/odds_cleaned.csv`: Cleaned betting odds data with fight IDs joined from `fights_cleaned.csv`, output by `odds_data_pre.ipynb`.

`/data/processed/random_forest_validation_predictions.csv`: Validation set predictions (actual label, predicted label, win probabilities, correct flag) from the base Random Forest model.

`/data/processed/scaler_exp1.pkl`: Fitted StandardScaler saved for the Experiment 1 feature set.

`/data/processed/scaler_exp2.pkl`: Fitted StandardScaler saved for the Experiment 2 feature set.

#### `/data/processed/Logistic_Regression_Outputs/`
Prediction CSVs produced by the logistic regression notebook for all experiments and splits.

`/data/processed/Logistic_Regression_Outputs/logistic_regression_validation_predictions.csv`: Logistic regression validation predictions for the base dataset.

`/data/processed/Logistic_Regression_Outputs/logistic_regression_validation_exp1_predictions.csv`: Logistic regression validation predictions for Experiment 1.

`/data/processed/Logistic_Regression_Outputs/logistic_regression_validation_exp2_predictions.csv`: Logistic regression validation predictions for Experiment 2.

`/data/processed/Logistic_Regression_Outputs/logistic_regression_test_predictions.csv`: Logistic regression test predictions for the base dataset.

`/data/processed/Logistic_Regression_Outputs/logistic_regression_test_exp1_predictions.csv`: Logistic regression test predictions for Experiment 1.

`/data/processed/Logistic_Regression_Outputs/logistic_regression_test_exp2_predictions.csv`: Logistic regression test predictions for Experiment 2.

#### `/data/scaler/`
Fitted scaler artifacts for the base feature set.

`/data/scaler/scaler.pkl`: Fitted StandardScaler for the base dataset feature set.

### `/models/`
Serialized trained model files saved with `joblib`.

`/models/logistic_regression_baseline.joblib`: Trained logistic regression model on the base dataset.

`/models/random_forest.joblib`: Best Random Forest model (tuned via `RandomizedSearchCV`) trained on the base dataset.

### `/notebooks/`
Jupyter notebooks implementing the full ML pipeline in order.

`/notebooks/data_cleaning.ipynb`: Loads and merges raw CSVs, handles missing values, normalizes fighter stats, and outputs `fights_cleaned.csv`.

`/notebooks/feature_engineering.ipynb`: Computes differential features between the two fighters in each matchup, applies scaling, and produces train/validation/test splits for all three experiments.

`/notebooks/odds_data_pre.ipynb`: Cleans `ufc_fights_with_betting_odds.csv`, joins fight IDs from the processed fight data by matching on fighter names and event date, and saves `odds_cleaned.csv`.

`/notebooks/logistic_regression.ipynb`: Trains a logistic regression baseline across all three experiments; evaluates with accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and ROC curves; saves prediction CSVs to `data/processed/Logistic_Regression_Outputs/`.

`/notebooks/model_random_forest.ipynb`: Trains a Random Forest classifier with `RandomizedSearchCV` hyperparameter tuning across all three experiments; evaluates with full metrics, feature importance plots, and accuracy-by-feature-bin plots; saves the base model to `models/random_forest.joblib` and validation predictions to `data/processed/`.

`/notebooks/model_xgboost.ipynb`: Trains an XGBoost classifier with Optuna hyperparameter optimization across Experiment 1 and Experiment 2; evaluates with accuracy, classification reports, confusion matrices, ROC curves, and feature importance plots.

### Root Files

`/requirements.txt`: Python package dependencies for the project.

`/.gitignore`: Specifies files and directories excluded from version control.

`/README.md`: This file — project overview, setup instructions, and directory/file descriptions.
