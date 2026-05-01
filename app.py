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

# ── Team jersey colors ─────────────────────────────────────
TEAM_COLORS = {
    "Mumbai Indians":                  {"primary": "#004BA0", "secondary": "#D1AB3E", "text": "#FFFFFF"},
    "Chennai Super Kings":             {"primary": "#F9CD05", "secondary": "#0081E9", "text": "#000000"},
    "Royal Challengers Bangalore":     {"primary": "#EC1C24", "secondary": "#000000", "text": "#FFFFFF"},
    "Kolkata Knight Riders":           {"primary": "#3A225D", "secondary": "#B3A123", "text": "#FFFFFF"},
    "Sunrisers Hyderabad":             {"primary": "#F7A721", "secondary": "#E8461A", "text": "#000000"},
    "Delhi Capitals":                  {"primary": "#0078BC", "secondary": "#EF1B23", "text": "#FFFFFF"},
    "Punjab Kings":                    {"primary": "#ED1B24", "secondary": "#A7A9AC", "text": "#FFFFFF"},
    "Rajasthan Royals":                {"primary": "#254AA5", "secondary": "#E8C5A0", "text": "#FFFFFF"},
    "Gujarat Titans":                  {"primary": "#1C2B4A", "secondary": "#C8A951", "text": "#FFFFFF"},
    "Lucknow Super Giants":            {"primary": "#A0DCFF", "secondary": "#FFCC00", "text": "#000000"},
}

# ── Custom CSS ─────────────────────────────────────────────
def inject_css(batting_team="Mumbai Indians"):
    tc = TEAM_COLORS.get(batting_team, {"primary": "#004BA0", "secondary": "#D1AB3E", "text": "#FFFFFF"})
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Rajdhani:wght@400;500;600;700&family=Barlow+Condensed:wght@400;600;700&display=swap');

    /* ── Background ── */
    .stApp {{
        background: radial-gradient(ellipse at 20% 10%, #0d1b2a 0%, #050e1a 60%, #0a0f1e 100%);
        background-attachment: fixed;
    }}

    /* Stadium lights effect */
    .stApp::before {{
        content: '';
        position: fixed;
        top: -200px;
        left: 50%;
        transform: translateX(-50%);
        width: 600px;
        height: 400px;
        background: radial-gradient(ellipse, rgba(255,220,80,0.08) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
    }}

    /* ── Main container ── */
    .main .block-container {{
        max-width: 760px;
        padding: 1rem 2rem 3rem;
    }}

    /* ── Typography ── */
    html, body, [class*="css"] {{
        font-family: 'Rajdhani', sans-serif;
        color: #e8eaf0;
    }}

    h1, h2, h3 {{
        font-family: 'Bebas Neue', sans-serif;
        letter-spacing: 2px;
    }}

    /* ── Header hero ── */
    .ipl-hero {{
        background: linear-gradient(135deg, {tc['primary']}22 0%, #0d1b2a 50%, {tc['secondary']}11 100%);
        border: 1px solid {tc['primary']}55;
        border-radius: 16px;
        padding: 28px 32px 20px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 40px {tc['primary']}30, inset 0 1px 0 rgba(255,255,255,0.05);
    }}

    .ipl-hero::before {{
        content: '🏟️';
        position: absolute;
        right: 24px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 80px;
        opacity: 0.07;
    }}

    .ipl-hero-title {{
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3rem;
        letter-spacing: 4px;
        color: #FFFFFF;
        line-height: 1;
        margin: 0;
        text-shadow: 0 0 30px {tc['primary']}aa;
    }}

    .ipl-hero-sub {{
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.95rem;
        letter-spacing: 3px;
        color: {tc['secondary']};
        text-transform: uppercase;
        margin-top: 6px;
        opacity: 0.9;
    }}

    .tata-badge {{
        display: inline-block;
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid {tc['secondary']}66;
        border-radius: 6px;
        padding: 3px 10px;
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.7rem;
        letter-spacing: 2px;
        color: {tc['secondary']};
        text-transform: uppercase;
        margin-top: 10px;
        box-shadow: 0 2px 8px {tc['secondary']}22;
    }}

    /* ── Team color banner ── */
    .team-banner {{
        background: linear-gradient(90deg, {tc['primary']}, {tc['secondary']}88);
        border-radius: 8px;
        padding: 10px 18px;
        margin: 14px 0;
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.15rem;
        letter-spacing: 2px;
        color: {tc['text']};
        box-shadow: 0 4px 16px {tc['primary']}44;
        border-left: 4px solid {tc['secondary']};
    }}

    /* ── Section cards ── */
    .input-card {{
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
        backdrop-filter: blur(4px);
    }}

    /* ── Streamlit widget overrides ── */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stRadio"] label {{
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.85rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #99aabb !important;
    }}

    div[data-testid="stSelectbox"] > div > div {{
        background: rgba(255,255,255,0.05);
        border: 1px solid {tc['primary']}55;
        border-radius: 8px;
        color: #e8eaf0;
    }}

    div[data-testid="stSelectbox"] > div > div:focus-within {{
        border-color: {tc['primary']};
        box-shadow: 0 0 0 2px {tc['primary']}44;
    }}

    /* Slider accent */
    div[data-testid="stSlider"] div[role="slider"] {{
        background: {tc['primary']} !important;
        border-color: {tc['secondary']} !important;
    }}

    div[data-testid="stSlider"] > div > div > div {{
        background: linear-gradient(90deg, {tc['primary']}, {tc['secondary']}) !important;
    }}

    /* Number input */
    div[data-testid="stNumberInput"] input {{
        background: rgba(255,255,255,0.05);
        border: 1px solid {tc['primary']}55;
        border-radius: 8px;
        color: #e8eaf0;
    }}

    /* ── Predict button ── */
    div[data-testid="stButton"] > button[kind="primary"] {{
        background: linear-gradient(135deg, {tc['primary']}, {tc['secondary']}) !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Bebas Neue', sans-serif !important;
        font-size: 1.3rem !important;
        letter-spacing: 3px !important;
        color: {tc['text']} !important;
        padding: 14px !important;
        box-shadow: 0 6px 24px {tc['primary']}55 !important;
        transition: all 0.2s ease !important;
    }}

    div[data-testid="stButton"] > button[kind="primary"]:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 32px {tc['primary']}77 !important;
    }}

    /* ── Result card ── */
    .result-hero {{
        background: linear-gradient(135deg, {tc['primary']}33, {tc['secondary']}11);
        border: 2px solid {tc['primary']}88;
        border-radius: 16px;
        padding: 28px 32px;
        margin: 16px 0;
        text-align: center;
        box-shadow: 0 0 60px {tc['primary']}33;
    }}

    .result-score {{
        font-family: 'Bebas Neue', sans-serif;
        font-size: 5rem;
        color: #FFFFFF;
        letter-spacing: 6px;
        line-height: 1;
        text-shadow: 0 0 40px {tc['primary']};
    }}

    .result-label {{
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.85rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: {tc['secondary']};
        margin-bottom: 8px;
    }}

    .result-range {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        color: rgba(255,255,255,0.55);
        margin-top: 8px;
    }}

    /* ── Stat chips ── */
    .stat-row {{
        display: flex;
        gap: 12px;
        justify-content: center;
        margin-top: 18px;
        flex-wrap: wrap;
    }}

    .stat-chip {{
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 50px;
        padding: 8px 18px;
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.9rem;
        letter-spacing: 1px;
        color: #ccd6f6;
    }}

    .stat-chip span {{
        color: {tc['secondary']};
        font-weight: 700;
    }}

    /* ── Commentary box ── */
    .commentary {{
        border-left: 3px solid {tc['secondary']};
        padding: 12px 18px;
        background: rgba(255,255,255,0.03);
        border-radius: 0 8px 8px 0;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.05rem;
        color: #ccd6f6;
        margin-top: 14px;
        letter-spacing: 0.3px;
    }}

    /* ── Divider ── */
    hr {{
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 20px 0;
    }}

    /* ── Footer ── */
    .ipl-footer {{
        text-align: center;
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.78rem;
        letter-spacing: 2px;
        color: rgba(255,255,255,0.25);
        text-transform: uppercase;
        margin-top: 32px;
    }}

    /* ── Error ── */
    div[data-testid="stAlert"] {{
        background: rgba(237,27,36,0.1) !important;
        border: 1px solid rgba(237,27,36,0.3) !important;
        border-radius: 10px !important;
    }}

    /* ── Radio buttons ── */
    div[data-testid="stRadio"] > div {{
        flex-direction: row;
        gap: 12px;
    }}

    div[data-testid="stRadio"] label {{
        background: rgba(255,255,255,0.04);
        border: 1px solid {tc['primary']}44;
        border-radius: 8px;
        padding: 8px 16px;
        cursor: pointer;
        font-size: 0.9rem !important;
        transition: all 0.2s;
    }}

    /* ── Metrics ── */
    div[data-testid="stMetric"] {{
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 12px 16px;
    }}

    div[data-testid="stMetricLabel"] {{
        font-family: 'Barlow Condensed', sans-serif !important;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #7788aa !important;
        font-size: 0.78rem !important;
    }}

    div[data-testid="stMetricValue"] {{
        font-family: 'Bebas Neue', sans-serif !important;
        font-size: 2rem !important;
        color: {tc['secondary']} !important;
    }}

    /* Hide Streamlit branding */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)


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

# ── Default CSS injection ──────────────────────────────────
if "batting_team" not in st.session_state:
    st.session_state.batting_team = TEAMS[0]

inject_css(st.session_state.get("batting_team", TEAMS[0]))

# ── Hero Header ───────────────────────────────────────────
tc = TEAM_COLORS.get(st.session_state.get("batting_team", TEAMS[0]), TEAM_COLORS[TEAMS[0]])

st.markdown(f"""
<div class="ipl-hero">
    <div style="display:flex; align-items:flex-start; justify-content:space-between;">
        <div>
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.7rem; letter-spacing:3px;
                        color:{tc['secondary']}; text-transform:uppercase; margin-bottom:4px;">
                ✦ TATA IPL · POWERED BY XGBOOST ✦
            </div>
            <div class="ipl-hero-title">🏏 IPL SCORE<br>PREDICTOR</div>
            <div class="ipl-hero-sub">Real-time final score forecast</div>
        </div>
        <div style="text-align:right; padding-top:6px;">
            <div style="font-family:'Bebas Neue',sans-serif; font-size:2.8rem; color:{tc['primary']};
                        line-height:1; letter-spacing:2px; text-shadow: 0 0 20px {tc['primary']};">
                TATA
            </div>
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.65rem; letter-spacing:3px;
                        color:rgba(255,255,255,0.3); text-transform:uppercase; margin-top:2px;">
                TITLE SPONSOR
            </div>
        </div>
    </div>
    <div style="margin-top:14px; height:3px; background:linear-gradient(90deg, {tc['primary']}, {tc['secondary']}, transparent);
                border-radius:2px;"></div>
</div>
""", unsafe_allow_html=True)

# ── Cricket Image Banner ──────────────────────────────────
st.markdown(f"""
<div style="position:relative; border-radius:14px; overflow:hidden; margin-bottom:22px;
            box-shadow: 0 8px 40px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.06);">
    <img src="https://images.unsplash.com/photo-1540747913346-19e32dc3e97e?w=900&q=80&auto=format&fit=crop"
         style="width:100%; height:200px; object-fit:cover; object-position:center 30%;
                display:block; filter:brightness(0.55) saturate(1.2);"
         alt="IPL Cricket Stadium" />
    <!-- gradient overlay -->
    <div style="position:absolute; inset:0;
                background:linear-gradient(to right, {tc['primary']}bb 0%, transparent 50%, {tc['secondary']}44 100%);"></div>
    <!-- left caption -->
    <div style="position:absolute; bottom:18px; left:20px;">
        <div style="font-family:'Bebas Neue',sans-serif; font-size:1.6rem; letter-spacing:3px;
                    color:#FFFFFF; text-shadow:0 2px 12px rgba(0,0,0,0.8); line-height:1;">
            THE BIGGEST CRICKET LEAGUE
        </div>
        <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.75rem; letter-spacing:2.5px;
                    color:{tc['secondary']}; text-transform:uppercase; margin-top:3px;">
            ⚡ Predict · Analyse · Dominate
        </div>
    </div>
    <!-- right badge -->
    <div style="position:absolute; top:14px; right:16px; background:rgba(0,0,0,0.55);
                border:1px solid {tc['secondary']}88; border-radius:6px; padding:4px 12px;
                font-family:'Barlow Condensed',sans-serif; font-size:0.7rem; letter-spacing:2px;
                color:{tc['secondary']}; backdrop-filter:blur(6px); text-transform:uppercase;">
        🏆 IPL 2008 – 2024
    </div>
</div>
""", unsafe_allow_html=True)

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
    bowling_team   = st.selectbox("🎳 Bowling Team", [t for t in TEAMS if t != batting_team])
    toss_batting   = st.radio("Did batting team win toss & chose to bat?", ["Yes", "No"])
    wickets_fallen = st.slider("Wickets Fallen", min_value=0, max_value=9, value=2)
    last5_runs     = st.number_input("Runs in Last 5 Overs", min_value=0, max_value=120, value=45)

# Update session state for color theming
st.session_state.batting_team = batting_team

# Team color banner
bat_tc = TEAM_COLORS.get(batting_team, TEAM_COLORS[TEAMS[0]])
bowl_tc = TEAM_COLORS.get(bowling_team, TEAM_COLORS[TEAMS[1]])
st.markdown(f"""
<div style="display:flex; gap:10px; margin:10px 0;">
    <div style="flex:1; background:linear-gradient(90deg, {bat_tc['primary']}, {bat_tc['secondary']}88);
                border-radius:8px; padding:10px 16px; font-family:'Bebas Neue',sans-serif;
                font-size:1rem; letter-spacing:2px; color:{bat_tc['text']};
                border-left:4px solid {bat_tc['secondary']}; box-shadow:0 4px 16px {bat_tc['primary']}44;">
        🏏 {batting_team}
    </div>
    <div style="display:flex; align-items:center; font-family:'Bebas Neue',sans-serif;
                font-size:1.2rem; color:rgba(255,255,255,0.4); padding:0 4px;">VS</div>
    <div style="flex:1; background:linear-gradient(90deg, {bowl_tc['primary']}, {bowl_tc['secondary']}88);
                border-radius:8px; padding:10px 16px; font-family:'Bebas Neue',sans-serif;
                font-size:1rem; letter-spacing:2px; color:{bowl_tc['text']};
                border-left:4px solid {bowl_tc['secondary']}; box-shadow:0 4px 16px {bowl_tc['primary']}44;">
        🎳 {bowling_team}
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── Predict ────────────────────────────────────────────────
if st.button("🔮 PREDICT FINAL SCORE", use_container_width=True, type="primary"):

    balls_bowled    = over * 6
    balls_remaining = max(0, 120 - balls_bowled)
    crr             = runs_so_far / over if over > 0 else 0
    toss_bat_flag   = 1 if toss_batting == "Yes" else 0

    def safe_encode(encoder, value):
        classes = list(encoder.classes_)
        if value in classes:
            return encoder.transform([value])[0]
        else:
            return 0

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
    low, high = prediction - 10, prediction + 10

    # ── Result card ──────────────────────────────────────
    st.markdown(f"""
    <div class="result-hero">
        <div class="result-label">⚡ PROJECTED FINAL SCORE</div>
        <div class="result-score">{prediction}</div>
        <div class="result-range">Expected range: <strong style="color:{bat_tc['secondary']}">{low} – {high}</strong> runs</div>
        <div class="stat-row">
            <div class="stat-chip">📊 CRR <span>{crr:.2f}</span></div>
            <div class="stat-chip">🎯 Overs Left <span>{20 - over}</span></div>
            <div class="stat-chip">🏏 Wickets Left <span>{10 - wickets_fallen}</span></div>
            <div class="stat-chip">⚡ L5 Runs <span>{last5_runs}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Commentary ────────────────────────────────────────
    if wickets_fallen <= 2 and crr >= 9:
        commentary = f"🔥 <strong>{batting_team}</strong> are absolutely on fire! Explosive batting with wickets in hand — the bowling attack has no answers tonight."
    elif wickets_fallen >= 6:
        commentary = f"⚠️ <strong>{batting_team}</strong> are in deep trouble with {wickets_fallen} wickets down. The lower order will need to dig in to post a competitive total."
    elif crr < 6:
        commentary = f"🐢 <strong>{batting_team}</strong> have started cautiously. The big overs are still to come — watch for acceleration in the death."
    elif last5_runs >= 60:
        commentary = f"🚀 <strong>{batting_team}</strong> have shifted gears brilliantly! {last5_runs} runs in the last 5 overs signals a blistering finish ahead."
    else:
        commentary = f"✅ <strong>{batting_team}</strong> are building a steady innings. Consistent pressure and smart cricket — looking good for a fighting total."

    st.markdown(f'<div class="commentary">{commentary}</div>', unsafe_allow_html=True)

    # Metrics row
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    r1.metric("🎯 Predicted Score", f"{prediction}")
    r2.metric("📈 Current Run Rate", f"{crr:.2f}")
    r3.metric("🏏 Wickets Left", f"{10 - wickets_fallen}")

# ── Footer ─────────────────────────────────────────────────
st.markdown("""
<div class="ipl-footer">
    <div style="margin-bottom:6px">
        ◆ &nbsp; TATA IPL SCORE PREDICTOR &nbsp; ◆
    </div>
    Built with XGBoost + Streamlit &nbsp;|&nbsp; Dataset: Kaggle IPL 2008–2020
</div>
""", unsafe_allow_html=True)
