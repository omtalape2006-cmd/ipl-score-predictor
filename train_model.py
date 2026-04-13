# ============================================================
# train_model.py — Train & evaluate the score prediction model
# ============================================================

import pandas as pd
import numpy as np
import pickle, os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from feature_engineering import get_feature_columns, encode_features, compute_innings_snapshots


def load_features(path="data/features.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "features.csv not found. Run feature_engineering.py first."
        )
    return pd.read_csv(path)


def train(df, model_type="xgboost"):
    FEATURES = get_feature_columns()
    TARGET   = "final_score"

    # Only train on snapshots at or after over 6 (more stable predictions)
    df = df[df["over"] >= 6].copy()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_type == "xgboost":
        model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
    else:
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

    print(f"Training {model_type} model on {len(X_train)} samples...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\n{'='*40}")
    print(f"  Model     : {model_type}")
    print(f"  MAE       : {mae:.2f} runs")
    print(f"  RMSE      : {rmse:.2f} runs")
    print(f"  R² Score  : {r2:.4f}")
    print(f"{'='*40}\n")

    return model, X_test, y_test, y_pred


def save_model(model, encoders, path="model/"):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "score_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(path, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    print(f"Model and encoders saved to {path}")


def load_model(path="model/"):
    with open(os.path.join(path, "score_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(path, "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    return model, encoders


def feature_importance(model, feature_cols):
    import matplotlib.pyplot as plt

    fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    fi.plot(kind="barh", ax=ax, color="#378ADD")
    ax.set_title("Feature Importance", fontsize=14)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("model/feature_importance.png", dpi=150)
    print("Feature importance chart saved to model/feature_importance.png")
    plt.show()


if __name__ == "__main__":
    df = load_features()

    # Re-encode (encoders needed for inference)
    df, encoders = encode_features(df)

    model, X_test, y_test, y_pred = train(df, model_type="xgboost")
    save_model(model, encoders)
    feature_importance(model, get_feature_columns())
