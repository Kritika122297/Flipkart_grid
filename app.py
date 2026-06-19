"""
🅿️ ParkWatch AI — Parking-Induced Congestion Intelligence Dashboard
Flipkart Gridlock Hackathon 2.0 — Round 2
Theme: Parking-Induced Congestion in Bengaluru
Run: streamlit run app.py
"""
import os
import streamlit as st
from config.styles import set_page_config, inject_css
from tabs import tab_data, tab_overview, tab_heatmap, tab_analytics, tab_enforcement, tab_impact, tab_simulator, tab_emergency, tab_timelapse, tab_ai, tab_compare
from data.loader import load_and_process_data

set_page_config()
inject_css()

# ── Header ──────────────────────────────────────────────────────────
st.markdown("""
<div style='
    padding: 28px 32px 20px;
    margin-bottom: 8px;
    background: linear-gradient(135deg,
        rgba(22,27,47,0.7) 0%,
        rgba(28,33,54,0.5) 100%);
    border: 1px solid rgba(99,102,241,0.14);
    border-radius: 18px;
    backdrop-filter: blur(20px);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
'>
    <!-- Left: branding -->
    <div>
        <div style='display:flex; align-items:center; gap:12px; margin-bottom:6px;'>
            <div style='
                background: linear-gradient(135deg, #818CF8, #38BDF8);
                border-radius: 12px;
                width: 44px; height: 44px;
                display: flex; align-items: center; justify-content: center;
                font-size: 1.4rem;
                box-shadow: 0 4px 14px rgba(129,140,248,0.4);
                flex-shrink: 0;
            '>🅿️</div>
            <h1 style='
                background: linear-gradient(90deg, #818CF8 0%, #38BDF8 60%, #A78BFA 100%);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                font-size: 2.2rem; font-weight: 900; margin: 0;
                letter-spacing: -0.03em; line-height: 1;
            '>ParkWatch AI</h1>
        </div>
        <p style='color:#94A3B8; font-size:0.95rem; margin:0 0 4px 56px; font-weight:400;'>
            AI-Powered Parking &amp; Congestion Intelligence for Bengaluru Traffic Police
        </p>
        <div style='margin-left:56px; display:flex; gap:10px; flex-wrap:wrap;'>
            <span style='
                background: rgba(129,140,248,0.12);
                border: 1px solid rgba(129,140,248,0.25);
                border-radius: 20px; padding: 2px 10px;
                color: #818CF8; font-size: 0.72rem; font-weight: 700;
                letter-spacing: 0.04em; text-transform: uppercase;
            '>Flipkart Gridlock Hackathon 2.0</span>
            <span style='
                background: rgba(52,211,153,0.1);
                border: 1px solid rgba(52,211,153,0.22);
                border-radius: 20px; padding: 2px 10px;
                color: #34D399; font-size: 0.72rem; font-weight: 700;
                letter-spacing: 0.04em; text-transform: uppercase;
            '>Round 2</span>
        </div>
    </div>
    <!-- Right: live stat chips -->
    <div style='display:flex; gap:10px; flex-wrap:wrap; align-items:center;'>
        <div style='
            background: rgba(22,27,47,0.8);
            border: 1px solid rgba(99,102,241,0.2);
            border-radius: 12px; padding: 10px 16px; text-align:center;
        '>
            <div style='color:#475569; font-size:0.64rem; text-transform:uppercase;
            letter-spacing:.1em; font-weight:700;'>Engine</div>
            <div style='color:#818CF8; font-size:0.88rem; font-weight:800;
            margin-top:2px;'>RandomForest · z-score</div>
        </div>
        <div style='
            background: rgba(22,27,47,0.8);
            border: 1px solid rgba(99,102,241,0.2);
            border-radius: 12px; padding: 10px 16px; text-align:center;
        '>
            <div style='color:#475569; font-size:0.64rem; text-transform:uppercase;
            letter-spacing:.1em; font-weight:700;'>City</div>
            <div style='color:#38BDF8; font-size:0.88rem; font-weight:800;
            margin-top:2px;'>Bengaluru 🇮🇳</div>
        </div>
        <div style='
            background: rgba(22,27,47,0.8);
            border: 1px solid rgba(99,102,241,0.2);
            border-radius: 12px; padding: 10px 16px; text-align:center;
        '>
            <div style='color:#475569; font-size:0.64rem; text-transform:uppercase;
            letter-spacing:.1em; font-weight:700;'>Tabs</div>
            <div style='color:#A78BFA; font-size:0.88rem; font-weight:800;
            margin-top:2px;'>11 modules</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Session state bootstrap ──────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.stats = None

# ── Process a pending upload (flag set by tab_data's on_change callback) ──
# The callback deliberately does NO disk I/O (it runs in the asyncio event
# loop thread and would block WebSocket heartbeats for a 100MB+ file write).
# Here we run in the script thread, so blocking I/O is safe.
if "_pending_upload_key" in st.session_state:
    _pending_key   = st.session_state.pop("_pending_upload_key")
    _uploaded_file = st.session_state.get("_file_uploader_widget")
    if _uploaded_file is not None:
        _tmp_path = os.path.join(os.path.dirname(__file__), "data", "_uploaded_data.csv")
        with open(_tmp_path, "wb") as _fh:
            _fh.write(_uploaded_file.getbuffer())
        try:
            _df, _stats = load_and_process_data(_tmp_path)
            st.session_state.df    = _df
            st.session_state.raw_df = _df.copy()
            st.session_state.stats = _stats
            st.session_state["_processed_upload"]  = _pending_key
            st.session_state["_cleaning_applied"]  = False
            st.session_state["_cleaning_changes"]  = []
        except Exception as _exc:
            st.session_state["_upload_error"] = str(_exc)

df = st.session_state.df
stats = st.session_state.stats

# ── Tabs ─────────────────────────────────────────────────────────────
def _no_data_msg(tab_hint: str = "") -> None:
    st.markdown(
        "<div class='pw-empty-state'>"
        "<div class='es-icon'>📂</div>"
        "<div class='es-title'>No data loaded yet</div>"
        "<div class='es-sub'>"
        "Go to the <b style='color:#6C63FF;'>📂 Data</b> tab and upload your BTP violation CSV to get started."
        + (f"<br><span style='color:#6C63FF;font-size:0.82rem;margin-top:6px;display:block;'>{tab_hint}</span>" if tab_hint else "")
        + "</div></div>",
        unsafe_allow_html=True,
    )

tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📂 Data",
    "🏠 Overview",
    "🗺️ Heatmap",
    "📊 Analytics",
    "🎯 Enforcement",
    "💰 Impact",
    "🎮 Simulator",
    "🚑 Emergency Impact",
    "⏳ Time-Lapse",
    "🤖 AI Insights",
    "📊 Compare",
])

with tab0:
    tab_data.render(df, stats)

with tab1:
    if df is not None:
        tab_overview.render(df)
    else:
        _no_data_msg("This tab shows KPI summary cards, violation trends, and station-level metrics.")

with tab2:
    if df is not None:
        tab_heatmap.render(df)
    else:
        _no_data_msg("This tab shows an interactive CIS-weighted heatmap across Bengaluru with filter controls.")

with tab3:
    if df is not None:
        tab_analytics.render(df)
    else:
        _no_data_msg("This tab shows deep-dive analytics: time patterns, vehicle breakdowns, and CIS distributions.")

with tab4:
    if df is not None:
        tab_enforcement.render(df)
    else:
        _no_data_msg("This tab ranks police stations by enforcement priority index and recommends patrol zones.")

with tab5:
    if df is not None:
        tab_impact.render(df)
    else:
        _no_data_msg("This tab estimates economic impact and congestion cost of parking violations.")

with tab6:
    if df is not None:
        tab_simulator.render(df)
    else:
        _no_data_msg("This tab lets you simulate what-if scenarios — e.g., 30% fewer violations in Koramangala.")

with tab7:
    if df is not None:
        tab_emergency.render(df)
    else:
        _no_data_msg("This tab calculates how parking violations affect ambulance response times to hospitals.")

with tab8:
    if df is not None:
        tab_timelapse.render(df)
    else:
        _no_data_msg("This tab animates 24-hour violation density patterns across the city.")

with tab9:
    if df is not None:
        tab_ai.render(df)
    else:
        _no_data_msg("This tab trains a RandomForest model on your data to predict high-risk hours and detect anomalies.")

with tab10:
    tab_compare.render()

# ── Footer ───────────────────────────────────────────────────────────
st.markdown(
    """<div style='
        text-align:center;
        padding: 32px 0 18px;
        margin-top: 48px;
        border-top: 1px solid rgba(99,102,241,0.12);
    '>
        <p style='color:#475569; font-size:0.76rem; letter-spacing:0.03em;'>
            🅿️ <b style='color:#818CF8;'>ParkWatch AI</b>
            &nbsp;·&nbsp; Flipkart Gridlock Hackathon 2.0 — Round 2
            &nbsp;·&nbsp; Powered by Streamlit · Plotly · Folium · scikit-learn
        </p>
    </div>""",
    unsafe_allow_html=True,
)
