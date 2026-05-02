import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="IPL Score Predictor", layout="wide")

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

# ---------------- THEME SWITCH ----------------
colA, colB = st.columns([6,1])

with colA:
    st.markdown("## 🏏 IPL Score Predictor")

with colB:
    theme = st.selectbox("Theme", ["Dark", "Light"], index=0)
    st.session_state.theme = theme

# ---------------- CSS THEMES ----------------
if st.session_state.theme == "Dark":
    bg = "#0f172a"
    text = "#ffffff"
    card = "#1e293b"
else:
    bg = "#f5f5f5"
    text = "#000000"
    card = "#ffffff"

st.markdown(f"""
<style>
body {{
    background-color: {bg};
    color: {text};
}}
.stApp {{
    background-color: {bg};
}}
.card {{
    background-color: {card};
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- NAVIGATION ----------------
menu = st.radio("Navigation", ["Home", "Predict"], horizontal=True)

# ---------------- HOME PAGE ----------------
if menu == "Home":

    st.markdown(f"""
    <div class="card">
        <h2>🏏 Welcome to IPL Score Predictor</h2>
        <p>
        This project predicts the final score of an IPL innings based on match conditions.
        </p>

        <h4>🔍 Features:</h4>
        <ul>
            <li>Real-time score prediction</li>
            <li>Machine Learning based model</li>
            <li>Interactive UI</li>
            <li>Dark/Light theme support</li>
        </ul>

        <h4>⚙️ Technologies Used:</h4>
        <ul>
            <li>Python</li>
            <li>Pandas & NumPy</li>
            <li>Scikit-learn / XGBoost</li>
            <li>Streamlit</li>
        </ul>

        <h4>📊 Dataset:</h4>
        IPL historical data (Kaggle)

        <h4>👨‍💻 Developed By:</h4>
        Om Talape
    </div>
    """ , unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("model/score_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_model()

# ---------------- PREDICT PAGE ----------------
if menu == "Predict":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    teams = [
        "Mumbai Indians","Chennai Super Kings","Royal Challengers Bangalore",
        "Kolkata Knight Riders","Sunrisers Hyderabad","Delhi Capitals",
        "Punjab Kings","Rajasthan Royals","Gujarat Titans","Lucknow Super Giants"
    ]

    venues = [
        "Wankhede Stadium","M Chinnaswamy Stadium","Eden Gardens",
        "MA Chidambaram Stadium","Rajiv Gandhi Stadium",
        "Arun Jaitley Stadium","Mohali Stadium","Jaipur Stadium",
        "Ahmedabad Stadium","DY Patil Stadium"
    ]

    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox("Batting Team", teams)
        venue = st.selectbox("Venue", venues)
        overs = st.slider("Overs", 6, 19, 10)
        runs = st.number_input("Runs", 0, 300, 80)

    with col2:
        bowling_team = st.selectbox("Bowling Team", teams)
        wickets = st.slider("Wickets", 0, 9, 2)
        last5 = st.number_input("Last 5 overs runs", 0, 100, 40)
        toss = st.radio("Toss decision", ["Yes", "No"])

    if st.button("Predict Score"):

        balls_remaining = 120 - overs * 6
        crr = runs / overs
        toss_val = 1 if toss == "Yes" else 0

        input_df = pd.DataFrame([{
            "over": overs,
            "runs_so_far": runs,
            "wickets_fallen": wickets,
            "balls_remaining": balls_remaining,
            "crr": crr,
            "last5_runs": last5,
            "toss_batting": toss_val,
            "batting_team_enc": encoders["batting_team"].transform([batting_team])[0],
            "bowling_team_enc": encoders["bowling_team"].transform([bowling_team])[0],
            "venue_enc": encoders["venue"].transform([venue])[0],
        }])

        prediction = int(model.predict(input_df)[0])

        st.success(f"Predicted Score: {prediction}")

    st.markdown('</div>', unsafe_allow_html=True)
