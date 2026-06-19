import streamlit as st


def set_page_config():
    st.set_page_config(
        page_title="ParkWatch AI — Bengaluru",
        page_icon="🅿️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def inject_css():
    st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
/* ── Root variables ── */
:root {
    --accent-1: #6C63FF;
    --accent-2: #00D2FF;
    --accent-3: #7C4DFF;
    --bg-dark: #0E1117;
    --card-bg: rgba(30, 33, 48, 0.65);
    --glass-border: rgba(108, 99, 255, 0.15);
    --text-primary: #E8E8E8;
    --text-muted: #888;
}
/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp {
    background: linear-gradient(135deg, #0E1117 0%, #1a1a2e 40%, #16213e 100%);
}
/* ── Hide Streamlit defaults ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
div[data-testid="stToolbar"] {display: none;}
div[data-testid="stDecoration"] {display: none;}
/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(30, 33, 48, 0.5);
    border-radius: 12px;
    padding: 6px;
    border: 1px solid rgba(108, 99, 255, 0.12);
}
.stTabs [data-baseweb="tab"] {
    height: 46px;
    border-radius: 10px;
    color: #888;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0 20px;
    background: transparent;
    border: none;
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(108,99,255,0.25), rgba(0,210,255,0.15)) !important;
    color: #fff !important;
    border: 1px solid rgba(108,99,255,0.3) !important;
    box-shadow: 0 0 15px rgba(108,99,255,0.15);
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}
.stTabs [data-baseweb="tab-border"] {
    display: none;
}
/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, rgba(108,99,255,0.18), rgba(0,210,255,0.10));
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg, #6C63FF, #00D2FF);
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(108,99,255,0.2);
}
.kpi-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6C63FF, #00D2FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.15;
}
.kpi-sub {
    font-size: 0.78rem;
    color: #666;
    margin-top: 4px;
}
/* ── Section headers ── */
.section-header {
    font-size: 1.6rem;
    font-weight: 700;
    color: #E8E8E8;
    margin-bottom: 4px;
}
.section-sub {
    font-size: 0.92rem;
    color: #666;
    margin-bottom: 20px;
}
/* ── Glass panels ── */
.glass-panel {
    background: rgba(30, 33, 48, 0.55);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(108,99,255,0.12);
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
}
/* ── Impact cards ── */
.impact-card {
    background: linear-gradient(135deg, rgba(255,75,75,0.12), rgba(255,165,0,0.08));
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,75,75,0.2);
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.impact-card .kpi-value {
    background: linear-gradient(135deg, #FF4B4B, #FFA500);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
/* ── Savings card ── */
.savings-card {
    background: linear-gradient(135deg, rgba(0,210,100,0.12), rgba(0,210,255,0.08));
    backdrop-filter: blur(16px);
    border: 1px solid rgba(0,210,100,0.2);
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.savings-card .kpi-value {
    background: linear-gradient(135deg, #00D264, #00D2FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(108,99,255,0.3); border-radius: 3px; }
/* ── Expander styling ── */
.streamlit-expanderHeader {
    font-weight: 600;
    color: #bbb !important;
    background: rgba(30, 33, 48, 0.5);
    border-radius: 10px;
}
/* ── Dataframe header ── */
div[data-testid="stDataFrame"] th {
    background: rgba(108,99,255,0.15) !important;
    color: #E8E8E8 !important;
    font-weight: 700 !important;
}
/* ── Plotly chart containers ── */
.stPlotlyChart {
    border-radius: 14px;
    overflow: hidden;
}
/* ── Centered loading overlay ── */
div[data-testid="stSpinner"] {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    background: rgba(14, 17, 23, 0.72) !important;
    backdrop-filter: blur(5px) !important;
    -webkit-backdrop-filter: blur(5px) !important;
    z-index: 99999 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 !important;
    padding: 0 !important;
}
div[data-testid="stSpinner"] > div {
    background: rgba(22, 25, 40, 0.98) !important;
    border: 1px solid rgba(108, 99, 255, 0.45) !important;
    border-radius: 22px !important;
    padding: 40px 56px !important;
    text-align: center !important;
    box-shadow: 0 0 0 1px rgba(108,99,255,0.1),
                0 24px 64px rgba(0,0,0,0.65),
                inset 0 1px 0 rgba(255,255,255,0.05) !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 18px !important;
    min-width: 240px !important;
}
div[data-testid="stSpinner"] p,
div[data-testid="stSpinner"] span {
    color: #d4d4e8 !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    margin: 0 !important;
}
div[data-testid="stSpinner"] svg {
    width: 52px !important;
    height: 52px !important;
    color: #6C63FF !important;
}
/* ── Info / Warning banners ── */
.pw-info-banner {
    background: linear-gradient(135deg, rgba(0,210,255,0.08), rgba(108,99,255,0.08));
    border: 1px solid rgba(0, 210, 255, 0.22);
    border-left: 4px solid #00D2FF;
    border-radius: 10px;
    padding: 13px 18px;
    margin: 10px 0 16px;
    color: #9ad8ef;
    font-size: 0.89rem;
    line-height: 1.55;
}
.pw-warn-banner {
    background: linear-gradient(135deg, rgba(255,165,0,0.09), rgba(255,75,75,0.06));
    border: 1px solid rgba(255, 165, 0, 0.28);
    border-left: 4px solid #FFA500;
    border-radius: 10px;
    padding: 13px 18px;
    margin: 10px 0 16px;
    color: #ffd080;
    font-size: 0.89rem;
    line-height: 1.55;
}
.pw-empty-state {
    text-align: center;
    padding: 80px 24px;
    background: rgba(30, 33, 48, 0.55);
    border: 1px dashed rgba(108, 99, 255, 0.2);
    border-radius: 16px;
    margin: 16px 0;
}
.pw-empty-state .es-icon { font-size: 2.8rem; margin-bottom: 14px; }
.pw-empty-state .es-title { color: #bbb; font-size: 1.05rem; font-weight: 700; margin-bottom: 8px; }
.pw-empty-state .es-sub { color: #555; font-size: 0.87rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)
