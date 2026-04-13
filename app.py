# ============================================================
# app.py — Streamlit IPL Score Predictor App
# Run with: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Score Predictor",
    page_icon="🏏",
    layout="centered"
)

# ── Load model ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model/score_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

try:
    model, encoders = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Team & Venue lists ─────────────────────────────────────
TEAMS = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Sunrisers Hyderabad", "Delhi Capitals",
    "Punjab Kings", "Rajasthan Royals", "Gujarat Titans", "Lucknow Super Giants"
]

VENUES = [
    "Wankhede Stadium", "M Chinnaswamy Stadium", "Eden Gardens",
    "MA Chidambaram Stadium", "Rajiv Gandhi International Cricket Stadium",
    "Arun Jaitley Stadium", "Punjab Cricket Association IS Bindra Stadium",
    "Sawai Mansingh Stadium", "Narendra Modi Stadium", "DY Patil Stadium"
]

# ── UI ─────────────────────────────────────────────────────
st.title("🏏 IPL Score Predictor")
st.markdown("Predict the **final innings score** based on current match situation.")
st.divider()

if not model_loaded:
    st.error(
        "⚠️ Model not found. Please run the training pipeline first:\n\n"
        "```\npython data_setup.py\npython feature_engineering.py\npython train_model.py\n```"
    )
    st.stop()

# ── Inputs ─────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("🏏 Batting Team", TEAMS)
    venue        = st.selectbox("🏟️ Venue", VENUES)
    over         = st.slider("Over Completed", min_value=6, max_value=19, value=10)
    runs_so_far  = st.number_input("Runs Scored So Far", min_value=0, max_value=250, value=80)

with col2:
    bowling_team    = st.selectbox("🎳 Bowling Team", [t for t in TEAMS if t != batting_team])
    toss_batting    = st.radio("Did batting team win toss & chose to bat?", ["Yes", "No"])
    wickets_fallen  = st.slider("Wickets Fallen", min_value=0, max_value=9, value=2)
    last5_runs      = st.number_input("Runs in Last 5 Overs", min_value=0, max_value=120, value=45)

st.divider()

# ── Predict ─────────────────────────────────────────────────
if st.button("🔮 Predict Final Score", use_container_width=True, type="primary"):

    balls_bowled    = over * 6
    balls_remaining = max(0, 120 - balls_bowled)
    crr             = runs_so_far / over if over > 0 else 0
    toss_bat_flag   = 1 if toss_batting == "Yes" else 0

    # Encode categorical features
    def safe_encode(encoder, value):
        classes = list(encoder.classes_)
        if value in classes:
            return encoder.transform([value])[0]
        else:
            return 0  # fallback for unseen label

    batting_enc = safe_encode(encoders["batting_team"], batting_team)
    bowling_enc = safe_encode(encoders["bowling_team"], bowling_team)
    venue_enc   = safe_encode(encoders["venue"], venue)

    input_df = pd.DataFrame([{
        "over":             over,
        "runs_so_far":      runs_so_far,
        "wickets_fallen":   wickets_fallen,
        "balls_remaining":  balls_remaining,
        "crr":              round(crr, 4),
        "last5_runs":       last5_runs,
        "toss_batting":     toss_bat_flag,
        "batting_team_enc": batting_enc,
        "bowling_team_enc": bowling_enc,
        "venue_enc":        venue_enc
    }])

    prediction = int(model.predict(input_df)[0])

    # ── Result display ──────────────────────────────────────
    st.markdown("### 📊 Prediction Result")

    r1, r2, r3 = st.columns(3)
    r1.metric("🎯 Predicted Score", f"{prediction}")
    r2.metric("📈 Current Run Rate", f"{crr:.2f}")
    r3.metric("🏏 Wickets Left", f"{10 - wickets_fallen}")

    # Confidence range (±10 runs)
    low, high = prediction - 10, prediction + 10
    st.info(f"📉 Likely range: **{low} – {high} runs** (±10 run confidence band)")

    # Pace commentary
    projected_crr = prediction / 20
    st.markdown("---")
    if wickets_fallen <= 2 and crr >= 9:
        st.success("🔥 Explosive start! Batting team is on fire.")
    elif wickets_fallen >= 6:
        st.warning("⚠️ Many wickets down. Score might be lower than projected.")
    elif crr < 6:
        st.info("🐢 Slow start. Team may accelerate in the death overs.")
    else:
        st.success("✅ Solid innings in progress.")

# ── Footer ──────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with XGBoost + Streamlit | Dataset: Kaggle IPL 2008–2020")
