# 🏏 IPL Score Predictor

Predict the final innings score of an IPL match using Machine Learning (XGBoost).

---

## Project Structure

```
ipl_score_predictor/
├── data/                   ← Place dataset CSVs here
├── model/                  ← Saved model files (auto-created)
├── plots/                  ← EDA charts (auto-created)
├── data_setup.py           ← Step 1: Load & merge data
├── feature_engineering.py  ← Step 2: Build features
├── eda.py                  ← (Optional) Exploratory analysis
├── train_model.py          ← Step 3: Train XGBoost model
├── app.py                  ← Step 4: Streamlit web app
└── requirements.txt
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
Go to: https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020

Download and place these two files in a `data/` folder:
- `IPL_matches.csv`
- `IPL_Ball_by_Ball.csv`

---

## Run the Pipeline

```bash
# Step 1 — Merge datasets
python data_setup.py

# Step 2 — Engineer features
python feature_engineering.py

# Step 3 (optional) — Explore the data
python eda.py

# Step 4 — Train the model
python train_model.py

# Step 5 — Launch the app
streamlit run app.py
```

---

## Expected Model Performance

| Metric | Value |
|--------|-------|
| MAE    | ~10–15 runs |
| RMSE   | ~13–18 runs |
| R²     | ~0.85–0.92 |

---

## Features Used

| Feature | Description |
|---------|-------------|
| over | Current over number |
| runs_so_far | Cumulative runs |
| wickets_fallen | Wickets lost |
| balls_remaining | Balls left in innings |
| crr | Current run rate |
| last5_runs | Runs in last 5 overs |
| toss_batting | Did batting team win toss & bat? |
| batting_team_enc | Encoded batting team |
| bowling_team_enc | Encoded bowling team |
| venue_enc | Encoded venue |
