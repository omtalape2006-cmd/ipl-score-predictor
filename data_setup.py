# ============================================================
# data_setup.py — Download & prepare IPL dataset
# ============================================================
# Dataset source: https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020
# Download manually from Kaggle and place these 2 files in a /data folder:
#   - IPL_matches.csv
#   - IPL_Ball_by_Ball.csv

import pandas as pd
import os

DATA_DIR = "data"

def load_data():
    matches_path = os.path.join(DATA_DIR, "IPL_matches.csv")
    balls_path   = os.path.join(DATA_DIR, "IPL_Ball_by_Ball.csv")

    if not os.path.exists(matches_path) or not os.path.exists(balls_path):
        raise FileNotFoundError(
            "Dataset files not found!\n"
            "Please download from Kaggle:\n"
            "https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020\n"
            "and place IPL_matches.csv and IPL_Ball_by_Ball.csv inside a 'data/' folder."
        )

    matches = pd.read_csv(matches_path)
    balls   = pd.read_csv(balls_path)
    print(f"Matches loaded:  {matches.shape}")
    print(f"Ball-by-ball loaded: {balls.shape}")
    return matches, balls


def merge_data(matches, balls):
    # Rename for consistency
    matches = matches.rename(columns={"id": "match_id"})

    # Merge on match_id
    df = balls.merge(
        matches[["match_id", "venue", "date", "season", "toss_winner", "toss_decision"]],
        on="match_id",
        how="left"
    )
    return df


if __name__ == "__main__":
    matches, balls = load_data()
    df = merge_data(matches, balls)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, "merged.csv"), index=False)
    print("Merged data saved to data/merged.csv")
