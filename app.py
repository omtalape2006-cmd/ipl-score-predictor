# ============================================================
# app.py — Premium Dark/Light Compatible IPL Score Predictor
# Run with: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import pickle

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="IPL Score Predictor",
    page_icon="🏏",
    layout="centered"
)

# ------------------------------------------------------------
# PREMIUM CSS (Dark Mode Fixed)
# ------------------------------------------------------------
st.markdown("""
<style>

/* Import Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

/* Full App */
.stApp {
    background: linear-gradient(135deg, #eef2f7, #d9e4f5);
    font-family: 'Poppins', sans-serif;
    color: #111111 !important;
}

/* Main Title */
h1 {
    color: #0f172a !important;
    text-align: center;
    font-size: 54px !important;
    font-weight: 800 !important;
    margin-bottom: 0px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #1e293b !important;
    margin-bottom: 30px;
}

/* Labels + Text */
label, p, div, span {
    color: #111111 !important;
}

/* Input Boxes */
[data-baseweb="select"] > div,
.stNumberInput > div > div,
.stSlider,
.stRadio {
    background: white !important;
    border-radius: 12px !important;
    color: black !important;
    border: 1px solid #dbeafe;
}

/* Select Text */
[data-baseweb="select"] * {
    color: black !important;
}

/* Number Input Text */
input {
    color: black !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    height: 52px;
    border: none;
    border-radius: 14px;
    background: linear-gradient(90deg, #0f172a, #2563eb);
    color: white !important;
    font-size: 18px;
    font-weight: 700;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background: white;
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 6px 14px rgba(0,0,0,0.08);
}

/* Success / Info */
.stAlert {
    border-radius: 12px;
}

/* Hide Footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# LOAD MODEL
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
except:
    st.error("Model files missing. Please train model first.")
    st.stop()

# ------------------------------------------------------------
# DATA
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
    "Punjab Cricket Association Stadium",
    "Sawai Mansingh Stadium",
    "Narendra Modi Stadium"
]

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown("<h1>IPL Score Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Predict the final innings score based on current match situation</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# ------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Batting Team", TEAMS)
    venue = st.selectbox("Venue", VENUES)
    over = st.slider("Overs Completed", 6, 19, 10)
    runs_so_far = st.number_input("Runs Scored So Far", 0, 250, 80)

with col2:
    bowling_team = st.selectbox(
        "Bowling Team",
        [team for team in TEAMS if team != batting_team]
    )

    toss_choice = st.radio(
        "Did batting team win toss & choose to bat?",
        ["Yes", "No"]
    )

    wickets = st.slider("Wickets Fallen", 0, 9, 2)
    last5 = st.number_input("Runs in Last 5 Overs", 0, 100, 45)

st.markdown("---")

# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------
if st.button("Predict Final Score"):

    balls_remaining = 120 - (over * 6)
    crr = runs_so_far / over
    toss_flag = 1 if toss_choice == "Yes" else 0

    def encode(enc, value):
        try:
            return enc.transform([value])[0]
        except:
            return 0

    input_df = pd.DataFrame([{
        "over": over,
        "runs_so_far": runs_so_far,
        "wickets_fallen": wickets,
        "balls_remaining": balls_remaining,
        "crr": crr,
        "last5_runs": last5,
        "toss_batting": toss_flag,
        "batting_team_enc": encode(encoders["batting_team"], batting_team),
        "bowling_team_enc": encode(encoders["bowling_team"], bowling_team),
        "venue_enc": encode(encoders["venue"], venue)
    }])

    prediction = int(model.predict(input_df)[0])

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------
    st.markdown("## Prediction Result")

    a, b, c = st.columns(3)

    a.metric("Predicted Score", prediction)
    b.metric("Run Rate", round(crr, 2))
    c.metric("Wickets Left", 10 - wickets)

    st.info(f"Likely Final Range: {prediction-10} to {prediction+10}")

    if wickets <= 2 and crr >= 9:
        st.success("Explosive innings in progress!")
    elif wickets >= 6:
        st.warning("Too many wickets lost. Final score may drop.")
    elif crr < 6:
        st.info("Slow innings so far. Death overs acceleration possible.")
    else:
        st.success("Balanced innings with strong finish possible.")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.markdown(
"""
<div style='text-align:center; color:#475569; font-size:15px;'>
Built with XGBoost + Streamlit | IPL Dataset (2008–2020)
</div>
""",
unsafe_allow_html=True
)
