# ============================================================
# app.py — Premium Streamlit IPL Score Predictor App
# Run with: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------------------------------------------------
# Page Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="IPL Score Predictor",
    layout="centered",
    page_icon="🏏"
)

# ------------------------------------------------------------
# Custom Styling
# ------------------------------------------------------------
st.markdown("""
<style>

/* Main App Background */
.stApp {
    background: linear-gradient(135deg, #f5f7fa, #dfe9f3);
    font-family: 'Segoe UI', sans-serif;
}

/* Title */
h1 {
    color: #0d1b2a;
    text-align: center;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 0;
}

/* Paragraph */
p {
    font-size: 18px;
    color: #333333;
}

/* Divider spacing */
hr {
    margin-top: 10px;
    margin-bottom: 20px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #1e3c72, #2a5298);
    color: white;
    border-radius: 10px;
    font-weight: bold;
    font-size: 16px;
    height: 3em;
    border: none;
    width: 100%;
}

/* Inputs */
.stSelectbox, .stNumberInput, .stSlider, .stRadio {
    background-color: white;
    border-radius: 10px;
    padding: 5px;
}

/* Metric cards */
[data-testid="metric-container"] {
    background-color: white;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
}

/* Hide Streamlit Footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Load Model
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Teams & Venues
# ------------------------------------------------------------
TEAMS = [
    "Mumbai Indians",
    "Chennai Super Kings",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Sunrisers Hyderabad",
    "Delhi Capitals",
    "Punjab Kings",
    "Rajasthan Royals",
    "Gujarat Titans",
    "Lucknow Super Giants"
]

VENUES = [
    "Wankhede Stadium",
    "M Chinnaswamy Stadium",
    "Eden Gardens",
    "MA Chidambaram Stadium",
    "Rajiv Gandhi International Cricket Stadium",
    "Arun Jaitley Stadium",
    "Punjab Cricket Association IS Bindra Stadium",
    "Sawai Mansingh Stadium",
    "Narendra Modi Stadium",
    "DY Patil Stadium"
]

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown("<h1>IPL Score Predictor</h1>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center;'>Predict the final innings score based on current match situation</p>",
    unsafe_allow_html=True
)

st.divider()

# ------------------------------------------------------------
# Error if model missing
# ------------------------------------------------------------
if not model_loaded:
    st.error(
        "Model files not found. Please run:\n\n"
        "python data_setup.py\n"
        "python feature_engineering.py\n"
        "python train_model.py"
    )
    st.stop()

# ------------------------------------------------------------
# Inputs
# ------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Batting Team", TEAMS)
    venue = st.selectbox("Venue", VENUES)
    over = st.slider("Over Completed", min_value=6, max_value=19, value=10)
    runs_so_far = st.number_input("Runs Scored So Far", min_value=0, max_value=250, value=80)

with col2:
    bowling_team = st.selectbox(
        "Bowling Team",
        [team for team in TEAMS if team != batting_team]
    )

    toss_batting = st.radio(
        "Did batting team win toss & choose to bat?",
        ["Yes", "No"]
    )

    wickets_fallen = st.slider("Wickets Fallen", min_value=0, max_value=9, value=2)
    last5_runs = st.number_input("Runs in Last 5 Overs", min_value=0, max_value=120, value=45)

st.divider()

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
if st.button("Predict Final Score", use_container_width=True):

    balls_bowled = over * 6
    balls_remaining = max(0, 120 - balls_bowled)
    crr = runs_so_far / over if over > 0 else 0
    toss_bat_flag = 1 if toss_batting == "Yes" else 0

    # Safe encoding
    def safe_encode(encoder, value):
        classes = list(encoder.classes_)
        if value in classes:
            return encoder.transform([value])[0]
        return 0

    batting_enc = safe_encode(encoders["batting_team"], batting_team)
    bowling_enc = safe_encode(encoders["bowling_team"], bowling_team)
    venue_enc = safe_encode(encoders["venue"], venue)

    input_df = pd.DataFrame([{
        "over": over,
        "runs_so_far": runs_so_far,
        "wickets_fallen": wickets_fallen,
        "balls_remaining": balls_remaining,
        "crr": round(crr, 4),
        "last5_runs": last5_runs,
        "toss_batting": toss_bat_flag,
        "batting_team_enc": batting_enc,
        "bowling_team_enc": bowling_enc,
        "venue_enc": venue_enc
    }])

    prediction = int(model.predict(input_df)[0])

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------
    st.markdown("## Match Prediction Result")

    r1, r2, r3 = st.columns(3)

    r1.metric("Predicted Score", prediction)
    r2.metric("Current Run Rate", f"{crr:.2f}")
    r3.metric("Wickets Left", 10 - wickets_fallen)

    low = prediction - 10
    high = prediction + 10

    st.info(f"Likely Range: {low} - {high} runs")

    st.markdown("---")

    # Commentary
    if wickets_fallen <= 2 and crr >= 9:
        st.success("Explosive start! Batting side is dominating.")
    elif wickets_fallen >= 6:
        st.warning("Too many wickets lost. Score may fall below expectation.")
    elif crr < 6:
        st.info("Slow start. Team may accelerate in death overs.")
    else:
        st.success("Balanced innings in progress.")

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")

st.markdown(
    "<p style='text-align:center; color:gray;'>Built using XGBoost + Streamlit | IPL Dataset (2008–2020)</p>",
    unsafe_allow_html=True
)
