# ============================================================
# eda.py — Exploratory Data Analysis
# Run this to understand your data before training
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("plots", exist_ok=True)
sns.set_theme(style="whitegrid")

df = pd.read_csv("data/features.csv")
print(f"Dataset shape: {df.shape}")
print(df.head())

# ── 1. Score distribution ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
df.drop_duplicates(["match_id", "inning"])["final_score"].hist(
    bins=40, ax=ax, color="#378ADD", edgecolor="white"
)
ax.set_title("Distribution of IPL Innings Final Scores")
ax.set_xlabel("Final Score")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("plots/score_distribution.png", dpi=150)
plt.show()

# ── 2. Average score by team ────────────────────────────────
team_scores = (
    df.drop_duplicates(["match_id", "inning"])
    .groupby("batting_team")["final_score"]
    .mean()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 5))
team_scores.plot(kind="bar", ax=ax, color="#1D9E75", edgecolor="white")
ax.set_title("Average Innings Score by Batting Team")
ax.set_xlabel("")
ax.set_ylabel("Avg Final Score")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("plots/avg_score_by_team.png", dpi=150)
plt.show()

# ── 3. Wickets vs Score ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
snap_at_15 = df[df["over"] == 15]
ax.scatter(snap_at_15["wickets_fallen"], snap_at_15["final_score"],
           alpha=0.3, color="#D85A30", s=15)
ax.set_title("Wickets Fallen (after 15 overs) vs Final Score")
ax.set_xlabel("Wickets Fallen")
ax.set_ylabel("Final Score")
plt.tight_layout()
plt.savefig("plots/wickets_vs_score.png", dpi=150)
plt.show()

# ── 4. Correlation heatmap ──────────────────────────────────
num_cols = ["over", "runs_so_far", "wickets_fallen", "crr",
            "last5_runs", "balls_remaining", "final_score"]
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f",
            cmap="Blues", ax=ax, square=True)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=150)
plt.show()

print("\nAll EDA plots saved to plots/")
