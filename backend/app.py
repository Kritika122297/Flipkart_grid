"""
🅿️ ParkWatch AI — Parking-Induced Congestion Intelligence Dashboard
Flipkart Gridlock Hackathon 2.0 — Round 2
Theme: Parking-Induced Congestion in Bengaluru
Run: streamlit run app.py
"""
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Groq API key — read once from environment, never exposed in the UI
_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
from config.styles import set_page_config, inject_css
from tabs import tab_data, tab_command_center, tab_congestion_analytics, tab_intelligent_dispatch, tab_tactical_commander, tab_cctv, tab_ortools_dispatch
from data.loader import load_and_process_data

set_page_config()
inject_css()

# ── Header ──────────────────────────────────────────────────────────
st.markdown("## 🅿️ ParkWatch AI")
st.caption("AI-Powered Parking Enforcement & Congestion Intelligence · Flipkart Gridlock Hackathon 2.0 · Bengaluru")
st.divider()

# ── Global chat state (persists across tab switches and reruns) ──────
if "chat_panel_open" not in st.session_state:
    st.session_state["chat_panel_open"] = False
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
st.session_state["groq_api_key"] = _GROQ_API_KEY

# ── Session state bootstrap (auto-load shared demo data if available) ────────
# Demo CSV is read-only shared data — safe to reference by path across sessions.
_DEMO_CSV = os.path.join(os.path.dirname(__file__), "data", "_demo_data.csv")
if "df" not in st.session_state:
    if os.path.exists(_DEMO_CSV):
        try:
            _df0, _stats0 = load_and_process_data(_DEMO_CSV)
            st.session_state.df     = _df0
            st.session_state.raw_df = _df0.copy()
            st.session_state.stats  = _stats0
        except Exception:
            st.session_state.df    = None
            st.session_state.stats = None
    else:
        st.session_state.df    = None
        st.session_state.stats = None

# ── Process a pending upload (flag set by tab_data's on_change callback) ──────
# Bytes are read from the UploadedFile object directly — no disk I/O, no shared
# static file, so concurrent users cannot overwrite each other's data.
if "_pending_upload_key" in st.session_state:
    _pending_key   = st.session_state.pop("_pending_upload_key")
    _uploaded_file = st.session_state.get("_file_uploader_widget")
    if _uploaded_file is not None:
        try:
            _csv_bytes = _uploaded_file.getvalue()          # bytes → hashable by st.cache_data
            _df, _stats = load_and_process_data(_csv_bytes)
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

# ── Sidebar: compact upload widget (no st.columns — sidebar restriction) ──
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:6px 0 14px;'>"
        "<span style='font-size:1.6rem;font-weight:900;color:#818CF8;'>🅿️ ParkWatch AI</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("##### 📂 Data Upload")
    tab_data._upload_widget()
    if df is not None:
        st.success(f"✅ {len(df):,} records loaded")
        st.caption("Full data panel → 📂 Data tab")
    else:
        st.info("Upload a BTP violation CSV to unlock the dashboard.")

# ── Tabs ─────────────────────────────────────────────────────────────
def no_data_state(tab_name: str, hint: str) -> None:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 📂 No data loaded")
        st.caption(f"Upload a CSV in the sidebar to activate **{tab_name}**.")
        st.info(hint)
        if st.button("Load demo data instead", key=f"demo_{tab_name}"):
            from data.loader import load_demo_data, load_and_process_data as _lap
            _ddf, _dstats = _lap(load_demo_data())
            st.session_state.df    = _ddf
            st.session_state.stats = _dstats
            st.rerun()
    st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Command Center",
    "📊 Congestion Analytics",
    "🎯 Intelligent Dispatch",
    "🤖 Tactical AI Commander",
    "📂 Data",
    "📷 Live CCTV",
    "🚛 OR-Tools Dispatch",
])

# Tab 1: Command Center
with tab1:
    if df is not None:
        tab_command_center.render(df)
    else:
        no_data_state(
            "Command Center",
            "Shows the live CIS-weighted heatmap, KPI cards, and 24-hour congestion animation.",
        )

# Tab 2: Congestion Analytics
with tab2:
    if df is not None:
        tab_congestion_analytics.render(df)
    else:
        no_data_state(
            "Congestion Analytics",
            "Deep-dive analytics, economic cost estimates, and emergency response analysis.",
        )

# Tab 3: Intelligent Dispatch
with tab3:
    if df is not None:
        tab_intelligent_dispatch.render(df)
    else:
        no_data_state(
            "Intelligent Dispatch",
            "EPI-ranked patrol routing, what-if simulation, and period comparison.",
        )

# Tab 4: Tactical AI Commander
with tab4:
    if df is not None:
        tab_tactical_commander.render(df)
    else:
        no_data_state(
            "Tactical AI Commander",
            "AI-powered patrol advice via Groq LLM, dispatch planning, and ML risk forecasting.",
        )

# Tab 5: Data (full upload, cleaning, EDA panel)
with tab5:
    tab_data.render(df, stats)

# Tab 6: Live CCTV Vision Pipeline (added by Vatsalya)
with tab6:
    tab_cctv.render(df)

# Tab 7: OR-Tools Fleet Dispatcher (added by Vatsalya)
with tab7:
    tab_ortools_dispatch.render(df)

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
