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

/* ══════════════════════════════════════
   SLATE DESIGN SYSTEM — CSS Variables
   ══════════════════════════════════════ */
:root {
    /* Palette */
    --bg-base:        #0D1117;
    --bg-elevated:    #161B27;
    --bg-surface:     #1C2136;
    --bg-surface-2:   #222840;

    /* Borders */
    --border-subtle:  rgba(99, 102, 241, 0.14);
    --border-medium:  rgba(99, 102, 241, 0.28);
    --border-strong:  rgba(99, 102, 241, 0.45);

    /* Accents */
    --accent-indigo:  #818CF8;
    --accent-sky:     #38BDF8;
    --accent-purple:  #A78BFA;
    --accent-green:   #34D399;
    --accent-red:     #F87171;
    --accent-amber:   #FBBF24;

    /* Text */
    --text-primary:   #E2E8F0;
    --text-secondary: #94A3B8;
    --text-faint:     #475569;

    /* Shadows */
    --shadow-sm:  0 2px 8px rgba(0,0,0,0.35);
    --shadow-md:  0 4px 20px rgba(0,0,0,0.45);
    --shadow-lg:  0 8px 40px rgba(0,0,0,0.55);
    --glow-indigo: 0 0 24px rgba(129,140,248,0.18);
}

/* ══════════════════════════════════════
   GLOBAL
   ══════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp {
    background: linear-gradient(160deg,
        var(--bg-base)      0%,
        var(--bg-elevated) 45%,
        var(--bg-surface)  100%);
    min-height: 100vh;
}

/* ── Hide Streamlit chrome ── */
#MainMenu              { visibility: hidden; }
footer                 { visibility: hidden; }
header                 { visibility: hidden; }
div[data-testid="stToolbar"]    { display: none; }
div[data-testid="stDecoration"] { display: none; }

/* ══════════════════════════════════════
   TAB BAR  (supports 11 tabs, wraps)
   ══════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    gap: 3px;
    background: rgba(22, 27, 47, 0.7);
    border-radius: 12px;
    padding: 5px;
    border: 1px solid var(--border-subtle);
    flex-wrap: wrap;
    backdrop-filter: blur(12px);
}
.stTabs [data-baseweb="tab"] {
    height: 38px;
    border-radius: 8px;
    color: var(--text-secondary);
    font-weight: 600;
    font-size: 0.78rem;
    padding: 0 11px;
    background: transparent;
    border: none;
    transition: all 0.2s ease;
    white-space: nowrap;
    letter-spacing: 0.01em;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background: rgba(129, 140, 248, 0.1) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,
        rgba(129,140,248,0.22),
        rgba(56,189,248,0.14)) !important;
    color: #fff !important;
    border: 1px solid var(--border-medium) !important;
    box-shadow: var(--glow-indigo);
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"]    { display: none; }

/* ══════════════════════════════════════
   KPI / METRIC CARDS
   ══════════════════════════════════════ */
.kpi-card {
    background: linear-gradient(145deg,
        rgba(28, 33, 54, 0.9),
        rgba(22, 27, 47, 0.85));
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 22px 18px 18px;
    text-align: center;
    box-shadow: var(--shadow-md), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 2px;
    background: linear-gradient(90deg, var(--accent-indigo), var(--accent-sky));
    opacity: 0.9;
}
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; right: 0;
    width: 80px; height: 80px;
    background: radial-gradient(circle at center,
        rgba(129,140,248,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-lg), var(--glow-indigo);
    border-color: var(--border-medium);
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1.4px;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent-indigo), var(--accent-sky));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.kpi-sub {
    font-size: 0.76rem;
    color: var(--text-faint);
    margin-top: 6px;
    font-weight: 500;
}

/* ══════════════════════════════════════
   SECTION HEADERS
   ══════════════════════════════════════ */
.section-header {
    font-size: 1.55rem;
    font-weight: 800;
    color: var(--text-primary);
    margin-bottom: 4px;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.section-sub {
    font-size: 0.88rem;
    color: var(--text-secondary);
    margin-bottom: 22px;
    line-height: 1.55;
    font-weight: 400;
}

/* ══════════════════════════════════════
   GLASS PANELS
   ══════════════════════════════════════ */
.glass-panel {
    background: rgba(22, 27, 47, 0.65);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 20px;
    box-shadow: var(--shadow-sm), inset 0 1px 0 rgba(255,255,255,0.03);
}

/* ══════════════════════════════════════
   IMPACT / SAVINGS CARDS
   ══════════════════════════════════════ */
.impact-card {
    background: linear-gradient(145deg,
        rgba(248, 113, 113, 0.1),
        rgba(251, 191, 36, 0.06));
    backdrop-filter: blur(16px);
    border: 1px solid rgba(248, 113, 113, 0.18);
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: var(--shadow-sm);
}
.impact-card .kpi-value {
    background: linear-gradient(135deg, var(--accent-red), var(--accent-amber));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.savings-card {
    background: linear-gradient(145deg,
        rgba(52, 211, 153, 0.1),
        rgba(56, 189, 248, 0.06));
    backdrop-filter: blur(16px);
    border: 1px solid rgba(52, 211, 153, 0.18);
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: var(--shadow-sm);
}
.savings-card .kpi-value {
    background: linear-gradient(135deg, var(--accent-green), var(--accent-sky));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ══════════════════════════════════════
   INFO / WARNING BANNERS
   ══════════════════════════════════════ */
.pw-info-banner {
    background: linear-gradient(135deg,
        rgba(56, 189, 248, 0.07),
        rgba(129, 140, 248, 0.07));
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-left: 3px solid var(--accent-sky);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 8px 0 16px;
    color: #9DD5F0;
    font-size: 0.87rem;
    line-height: 1.55;
}
.pw-warn-banner {
    background: linear-gradient(135deg,
        rgba(251, 191, 36, 0.08),
        rgba(248, 113, 113, 0.05));
    border: 1px solid rgba(251, 191, 36, 0.24);
    border-left: 3px solid var(--accent-amber);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 8px 0 16px;
    color: #FCD37A;
    font-size: 0.87rem;
    line-height: 1.55;
}
.pw-empty-state {
    text-align: center;
    padding: 72px 24px;
    background: rgba(22, 27, 47, 0.5);
    border: 1px dashed rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    margin: 16px 0;
}
.pw-empty-state .es-icon  { font-size: 2.6rem; margin-bottom: 14px; }
.pw-empty-state .es-title { color: var(--text-primary); font-size: 1rem;
                            font-weight: 700; margin-bottom: 8px; }
.pw-empty-state .es-sub   { color: var(--text-faint); font-size: 0.86rem;
                            line-height: 1.6; }

/* ══════════════════════════════════════
   HEATMAP STAT PILLS
   ══════════════════════════════════════ */
.stat-pill {
    background: rgba(22, 27, 47, 0.75);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    backdrop-filter: blur(12px);
    padding: 16px 12px 14px;
    text-align: center;
    margin-bottom: 6px;
    transition: border-color 0.2s;
}
.stat-pill:hover { border-color: var(--border-medium); }
.stat-pill .sp-icon  { font-size: 1.2rem; }
.stat-pill .sp-label {
    color: var(--text-secondary);
    font-size: 0.68rem;
    letter-spacing: .09em;
    text-transform: uppercase;
    margin: 5px 0 3px;
    font-weight: 600;
}
.stat-pill .sp-val {
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, var(--accent-indigo), var(--accent-sky));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ══════════════════════════════════════
   HEATMAP HOTSPOT LIST
   ══════════════════════════════════════ */
.hmap-section-hd {
    color: var(--text-secondary);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    margin: 20px 0 10px;
}
.hotspot-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 12px;
    margin-bottom: 6px;
    background: rgba(22, 27, 47, 0.65);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    backdrop-filter: blur(12px);
    transition: border-color 0.2s, background 0.2s;
}
.hotspot-item:hover {
    border-color: var(--border-medium);
    background: rgba(28, 33, 54, 0.8);
}
.hs-rank {
    min-width: 24px; height: 24px; line-height: 24px; text-align: center;
    border-radius: 50%; font-size: 0.7rem; font-weight: 800;
    background: linear-gradient(135deg, var(--accent-indigo), var(--accent-sky));
    color: #fff; flex-shrink: 0;
}
.hs-addr  { flex: 1; color: var(--text-primary); font-size: 0.8rem; line-height: 1.3; }
.hs-stats { text-align: right; white-space: nowrap; flex-shrink: 0; }
.hs-count { color: var(--accent-indigo); font-weight: 700; font-size: 0.84rem; }
.hs-cis   { color: var(--text-faint); font-size: 0.69rem; margin-top: 2px; }

/* ══════════════════════════════════════
   MAP LEGEND
   ══════════════════════════════════════ */
.map-legend {
    margin-top: 14px;
    padding: 10px 16px;
    background: rgba(22, 27, 47, 0.7);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    backdrop-filter: blur(12px);
}

/* ══════════════════════════════════════
   SCROLLBAR
   ══════════════════════════════════════ */
::-webkit-scrollbar       { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(129,140,248,0.28); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(129,140,248,0.5); }

/* ══════════════════════════════════════
   EXPANDERS
   ══════════════════════════════════════ */
.streamlit-expanderHeader {
    font-weight: 600;
    color: var(--text-secondary) !important;
    background: rgba(22, 27, 47, 0.6) !important;
    border-radius: 10px !important;
    border: 1px solid var(--border-subtle) !important;
}
.streamlit-expanderHeader:hover {
    color: var(--text-primary) !important;
    border-color: var(--border-medium) !important;
}

/* ══════════════════════════════════════
   DATA TABLE
   ══════════════════════════════════════ */
div[data-testid="stDataFrame"] th {
    background: rgba(129,140,248,0.12) !important;
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
div[data-testid="stDataFrame"] td {
    color: var(--text-secondary) !important;
    font-size: 0.83rem !important;
}

/* ══════════════════════════════════════
   PLOTLY CHART CONTAINERS
   ══════════════════════════════════════ */
.stPlotlyChart {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid var(--border-subtle);
    background: rgba(22, 27, 47, 0.4);
}

/* ══════════════════════════════════════
   BUTTONS
   ══════════════════════════════════════ */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent-indigo), #6366F1) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    transition: opacity 0.2s, transform 0.2s !important;
    box-shadow: 0 4px 14px rgba(129,140,248,0.35) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stDownloadButton"] > button {
    background: rgba(22, 27, 47, 0.8) !important;
    border: 1px solid var(--border-medium) !important;
    color: var(--accent-sky) !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all 0.2s !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    background: rgba(56,189,248,0.1) !important;
    border-color: var(--accent-sky) !important;
}

/* ══════════════════════════════════════
   FORM / INPUTS
   ══════════════════════════════════════ */
div[data-baseweb="select"] > div,
div[data-baseweb="input"]  > div {
    background: rgba(22, 27, 47, 0.7) !important;
    border-color: var(--border-subtle) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}
.stSlider [data-baseweb="slider"] {
    background: var(--border-subtle) !important;
}

/* ══════════════════════════════════════
   CENTERED LOADING OVERLAY
   ══════════════════════════════════════ */
div[data-testid="stSpinner"] {
    position: fixed !important;
    top: 0 !important; left: 0 !important;
    right: 0 !important; bottom: 0 !important;
    width: 100vw !important; height: 100vh !important;
    background: rgba(13, 17, 23, 0.78) !important;
    backdrop-filter: blur(6px) !important;
    -webkit-backdrop-filter: blur(6px) !important;
    z-index: 99999 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 !important; padding: 0 !important;
}
div[data-testid="stSpinner"] > div {
    background: rgba(22, 27, 47, 0.98) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 20px !important;
    padding: 38px 54px !important;
    text-align: center !important;
    box-shadow: 0 0 0 1px rgba(129,140,248,0.08),
                0 24px 60px rgba(0,0,0,0.65),
                inset 0 1px 0 rgba(255,255,255,0.04) !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 18px !important;
    min-width: 240px !important;
}
div[data-testid="stSpinner"] p,
div[data-testid="stSpinner"] span {
    color: var(--text-primary) !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    margin: 0 !important;
}
div[data-testid="stSpinner"] svg {
    width: 50px !important; height: 50px !important;
    color: var(--accent-indigo) !important;
}
</style>
""", unsafe_allow_html=True)
