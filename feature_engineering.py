# ============================================================
# feature_engineering.py — Build model-ready features
# ============================================================

import pandas as pd
import numpy as np

# Teams that have been consistently active — used for encoding
TEAMS = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Sunrisers Hyderabad", "Delhi Capitals",
    "Punjab Kings", "Rajasthan Royals", "Gujarat Titans", "Lucknow Super Giants"
]


def compute_innings_snapshots(df):
    """
    For each match+innings, create a snapshot at every over boundary.
    Returns a DataFrame where each row = state at end of a given over.
    """
    records = []

    # Group by match and innings
    grouped = df.groupby(["match_id", "inning"])

    for (match_id, inning), group in grouped:
        group = group.sort_values("over")

        batting_team = group["batting_team"].iloc[0]
        bowling_team = group["bowling_team"].iloc[0]
        venue        = group["venue"].iloc[0]
        season       = group["season"].iloc[0]
        toss_winner  = group["toss_winner"].iloc[0]
        toss_decision= group["toss_decision"].iloc[0]

        # Final score for this innings
        final_score = group["total_runs"].sum()

        # Cumulative stats over overs
        cumulative_runs    = 0
        cumulative_wickets = 0

        over_runs = []

        for over_num in sorted(group["over"].unique()):
            over_data = group[group["over"] == over_num]
            runs_this_over    = over_data["total_runs"].sum()
            wickets_this_over = over_data["is_wicket"].sum()

            cumulative_runs    += runs_this_over
            cumulative_wickets += wickets_this_over
            over_runs.append(runs_this_over)

            overs_completed = over_num  # 0-indexed, so over 0 = after 1st over
            balls_bowled    = (over_num + 1) * 6
            balls_remaining = max(0, 120 - balls_bowled)
            crr             = cumulative_runs / (over_num + 1) if over_num >= 0 else 0

            # Last 5 overs momentum
            last5_runs = sum(over_runs[-5:]) if len(over_runs) >= 5 else sum(over_runs)

            # Toss advantage flag
            toss_batting = 1 if (toss_winner == batting_team and toss_decision == "bat") else 0

            records.append({
                "match_id":         match_id,
                "inning":           inning,
                "batting_team":     batting_team,
                "bowling_team":     bowling_team,
                "venue":            venue,
                "season":           season,
                "over":             over_num + 1,       # 1-indexed for readability
                "runs_so_far":      cumulative_runs,
                "wickets_fallen":   cumulative_wickets,
                "balls_remaining":  balls_remaining,
                "crr":              round(crr, 4),
                "last5_runs":       last5_runs,
                "toss_batting":     toss_batting,
                "final_score":      final_score
            })

    return pd.DataFrame(records)


def encode_features(df):
    """Label encode categorical columns."""
    from sklearn.preprocessing import LabelEncoder

    cat_cols = ["batting_team", "bowling_team", "venue"]
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def get_feature_columns():
    return [
        "over", "runs_so_far", "wickets_fallen", "balls_remaining",
        "crr", "last5_runs", "toss_batting",
        "batting_team_enc", "bowling_team_enc", "venue_enc"
    ]


if __name__ == "__main__":
    import os
    df_raw = pd.read_csv("data/merged.csv")
    print("Building snapshots (this may take a minute)...")
    df_snap = compute_innings_snapshots(df_raw)
    df_snap, _ = encode_features(df_snap)
    os.makedirs("data", exist_ok=True)
    df_snap.to_csv("data/features.csv", index=False)
    print(f"Features saved. Shape: {df_snap.shape}")
    print(df_snap.head())
