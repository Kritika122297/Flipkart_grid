"""
🅿️ ParkWatch AI — Parking-Induced Congestion Intelligence Dashboard
Flipkart Gridlock Hackathon 2.0 — Round 2
Theme: Parking-Induced Congestion in Bengaluru
Run: streamlit run app.py
"""
import os
import streamlit as st
from config.styles import set_page_config, inject_css
from tabs import tab_data, tab_overview, tab_heatmap, tab_analytics, tab_enforcement, tab_impact, tab_simulator, tab_emergency, tab_timelapse
from data.loader import load_and_process_data

set_page_config()
inject_css()

# ── Header ──────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0;'>
    <h1 style='background: linear-gradient(90deg, #6C63FF, #00D2FF);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               font-size: 2.8rem; font-weight: 800; margin-bottom: 0;'>
        🅿️ ParkWatch AI
    </h1>
    <p style='color: #888; font-size: 1.1rem; margin-top: 5px;'>
        AI-Powered Parking Intelligence for Bengaluru Traffic Police
    </p>
    <p style='color: #555; font-size: 0.8rem;'>
        Flipkart Gridlock Hackathon 2.0 — Round 2
    </p>
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
_NO_DATA_MSG = (
    "<div class='glass-panel' style='text-align:center; padding:60px 20px;'>"
    "<div style='font-size:2.5rem; margin-bottom:12px;'>📂</div>"
    "<div style='color:#bbb; font-size:1.1rem; font-weight:600;'>No data loaded yet</div>"
    "<div style='color:#666; font-size:0.88rem; margin-top:8px;'>Go to the <b style='color:#6C63FF;'>📂 Data</b> tab and upload your CSV to get started.</div>"
    "</div>"
)

tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📂 Data",
    "🏠 Overview",
    "🗺️ Heatmap",
    "📊 Analytics",
    "🎯 Enforcement",
    "💰 Impact",
    "🎮 Simulator",
    "🚑 Emergency Impact",
    "⏳ Time-Lapse",
])

with tab0:
    tab_data.render(df, stats)

with tab1:
    if df is not None:
        tab_overview.render(df)
    else:
        st.markdown(_NO_DATA_MSG, unsafe_allow_html=True)

with tab2:
    if df is not None:
        tab_heatmap.render(df)
    else:
        st.markdown(_NO_DATA_MSG, unsafe_allow_html=True)

with tab3:
    if df is not None:
        tab_analytics.render(df)
    else:
        st.markdown(_NO_DATA_MSG, unsafe_allow_html=True)

with tab4:
    if df is not None:
        tab_enforcement.render(df)
    else:
        st.markdown(_NO_DATA_MSG, unsafe_allow_html=True)

with tab5:
    if df is not None:
        tab_impact.render(df)
    else:
        st.markdown(_NO_DATA_MSG, unsafe_allow_html=True)

with tab6:
    if df is not None:
        tab_simulator.render(df)
    else:
        st.markdown(_NO_DATA_MSG, unsafe_allow_html=True)

with tab7:
    if df is not None:
        tab_emergency.render(df)
    else:
        st.markdown(_NO_DATA_MSG, unsafe_allow_html=True)

with tab8:
    if df is not None:
        tab_timelapse.render(df)
    else:
        st.markdown(_NO_DATA_MSG, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────
st.markdown(
    """<div style='text-align:center; padding: 40px 0 20px 0; border-top: 1px solid rgba(108,99,255,0.1); margin-top: 40px;'>
        <p style='color:#444; font-size:0.78rem;'>
            Built with ❤️ for Flipkart Gridlock Hackathon 2.0 — ParkWatch AI
            &nbsp;|&nbsp; Powered by Streamlit, Plotly & Folium
        </p>
    </div>""",
    unsafe_allow_html=True,
)
