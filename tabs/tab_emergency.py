"""
tabs/tab_emergency.py — 🚑 Emergency Impact Tab
Shows how illegal parking delays ambulances and costs lives.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from math import radians, sin, cos, sqrt, atan2

# ── Constants ─────────────────────────────────────────────────────────
DAILY_AMBULANCE_CALLS     = 85
MAX_PARKING_DELAY_MIN     = 8.0
VALUE_OF_TIME_RS_PER_MIN  = 3.33
CARDIAC_SURVIVAL_DROP     = 0.10
AMBULANCE_SPEED_KMPH      = 40

HOSPITALS = [
    {"name": "St. John's Hospital",               "lat": 12.9547, "lon": 77.6168},
    {"name": "Narayana Health",                    "lat": 12.8998, "lon": 77.6027},
    {"name": "Manipal Hospital",                   "lat": 12.9525, "lon": 77.6476},
    {"name": "NIMHANS",                            "lat": 12.9398, "lon": 77.5962},
    {"name": "Victoria Hospital",                  "lat": 12.9659, "lon": 77.5749},
    {"name": "Bowring Hospital",                   "lat": 12.9763, "lon": 77.6074},
    {"name": "Apollo Hospital (Jayanagar)",        "lat": 12.9249, "lon": 77.5826},
    {"name": "Fortis Hospital (Bannerghatta)",     "lat": 12.8726, "lon": 77.5975},
]

_COLOR_MAP = {
    "Low (<2 min)":       "#10B981",
    "Moderate (2-4 min)": "#F59E0B",
    "High (4-6 min)":     "#EF4444",
    "Critical (>6 min)":  "#7C3AED",
}


# ── Helpers ───────────────────────────────────────────────────────────
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ── Cached Aggregation ────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_station_risk(_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        _df.groupby("police_station")
        .agg(
            avg_cis=("cis", "mean"),
            count=("cis", "size"),
            avg_lat=("latitude", "mean"),
            avg_lon=("longitude", "mean"),
        )
        .reset_index()
    )
    agg = agg.dropna(subset=["avg_lat", "avg_lon"])
    agg["est_delay"] = agg["avg_cis"] / 100 * MAX_PARKING_DELAY_MIN
    agg["delay_category"] = pd.cut(
        agg["est_delay"],
        bins=[0, 2, 4, 6, 8.01],
        labels=["Low (<2 min)", "Moderate (2-4 min)", "High (4-6 min)", "Critical (>6 min)"],
        include_lowest=True,
    ).astype(str)
    return agg


# ── Section 1: Hero Banner ────────────────────────────────────────────
def _render_hero():
    st.markdown("""
<div style="background:linear-gradient(135deg,rgba(239,68,68,0.15),
rgba(251,146,60,0.10)); border:1px solid rgba(239,68,68,0.25);
border-radius:20px; padding:32px; text-align:center; margin-bottom:24px;">
  <div style="font-size:3rem; margin-bottom:8px;">🚑</div>
  <h1 style="background:linear-gradient(90deg,#EF4444,#F97316);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  font-size:2rem; font-weight:800; margin:0;">
    Every Minute Matters
  </h1>
  <p style="color:#aaa; font-size:1.05rem; margin-top:8px;">
    Illegal parking in Bengaluru delays ambulances by up to 8 minutes.
    In cardiac emergencies, each minute reduces survival by 10%.
  </p>
</div>
""", unsafe_allow_html=True)


# ── Section 2: City-Wide Impact Metrics ──────────────────────────────
def _render_city_metrics(df: pd.DataFrame, station_agg: pd.DataFrame):
    avg_cis_city      = df["cis"].mean()
    avg_delay         = avg_cis_city / 100 * MAX_PARKING_DELAY_MIN
    critical_zones    = int((station_agg["avg_cis"] > 60).sum())
    peak_delay_hour   = int(df.groupby("hour")["cis"].mean().idxmax())
    annual_delays     = avg_delay * DAILY_AMBULANCE_CALLS * 365
    annual_delay_hrs  = annual_delays * DAILY_AMBULANCE_CALLS / 60

    cards = [
        ("⏱️ Avg Delay Per Call",  f"{avg_delay:.1f} min",        "due to parking"),
        ("🔴 Critical Risk Zones", str(critical_zones),             "stations with CIS > 60"),
        ("⚡ Peak Delay Hour",     f"{peak_delay_hour}:00",        "highest avg CIS"),
        ("📅 Annual Delay Hours",  f"{annual_delay_hrs:,.0f}",     "city-wide estimate"),
    ]

    cols = st.columns(4)
    for col, (title, value, sub) in zip(cols, cards):
        col.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(239,68,68,0.15),rgba(251,146,60,0.08));
border:1px solid rgba(239,68,68,0.25); border-radius:14px; padding:20px 16px;
text-align:center;">
  <div style="font-size:0.82rem; color:#aaa; margin-bottom:6px;">{title}</div>
  <div style="background:linear-gradient(135deg,#EF4444,#F97316);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  font-size:2rem; font-weight:800; line-height:1.1;">{value}</div>
  <div style="font-size:0.75rem; color:#666; margin-top:4px;">{sub}</div>
</div>
""", unsafe_allow_html=True)


# ── Section 3: Interactive Risk Map ───────────────────────────────────
def render_risk_map(station_agg: pd.DataFrame):
    st.markdown("### 🗺️ Emergency Response Risk Map")
    st.markdown(
        "<p style='color:#888; font-size:0.88rem; margin-top:-8px; margin-bottom:12px;'>"
        "Bubble size&nbsp;=&nbsp;violation count &nbsp;|&nbsp; "
        "Color&nbsp;=&nbsp;estimated ambulance delay</p>",
        unsafe_allow_html=True,
    )

    fig = px.scatter_mapbox(
        station_agg,
        lat="avg_lat",
        lon="avg_lon",
        color="delay_category",
        size="count",
        size_max=30,
        hover_name="police_station",
        hover_data={
            "est_delay":      ":.2f",
            "avg_cis":        ":.1f",
            "count":          True,
            "avg_lat":        False,
            "avg_lon":        False,
            "delay_category": False,
        },
        color_discrete_map=_COLOR_MAP,
        category_orders={"delay_category": list(_COLOR_MAP.keys())},
        mapbox_style="carto-darkmatter",
        center={"lat": 12.9716, "lon": 77.5946},
        zoom=11,
        height=520,
        labels={
            "delay_category": "Delay Category",
            "est_delay":      "Est. Delay (min)",
            "avg_cis":        "Avg CIS",
        },
    )

    hospital_df = pd.DataFrame(HOSPITALS)
    fig.add_trace(go.Scattermapbox(
        lat=hospital_df["lat"],
        lon=hospital_df["lon"],
        mode="markers+text",
        marker=dict(size=14, color="#FF6B9D", symbol="hospital"),
        text=hospital_df["name"],
        textposition="top right",
        textfont=dict(color="white", size=10),
        name="🏥 Hospitals",
        hovertemplate="<b>%{text}</b><extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            bgcolor="rgba(20,20,30,0.8)",
            bordercolor="rgba(108,99,255,0.2)",
            borderwidth=1,
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Section 4: Interactive Delay Calculator ───────────────────────────
def render_delay_calculator(df: pd.DataFrame, station_agg: pd.DataFrame):
    st.markdown("### 🧮 Response Time Calculator")
    st.markdown(
        "<hr style='border:1px solid rgba(255,255,255,0.07);'>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])

    with left:
        stations = sorted(df["police_station"].dropna().unique())
        origin      = st.selectbox("📍 Incident Zone (Police Station)", stations, key="emg_origin")
        destination = st.selectbox(
            "🏥 Nearest Hospital", [h["name"] for h in HOSPITALS], key="emg_dest"
        )
        calc_clicked = st.button(
            "🔍 Calculate Response Time", use_container_width=True, key="emg_calc_btn"
        )

    if calc_clicked:
        st.session_state["_emg_origin"] = origin
        st.session_state["_emg_dest"]   = destination
        st.session_state["_emg_done"]   = True

    with right:
        if st.session_state.get("_emg_done"):
            _origin = st.session_state["_emg_origin"]
            _dest   = st.session_state["_emg_dest"]

            station_row = station_agg[station_agg["police_station"] == _origin]
            hosp        = next((h for h in HOSPITALS if h["name"] == _dest), None)

            if station_row.empty or hosp is None:
                st.warning("No location data available for the selected station.")
            else:
                slat           = float(station_row["avg_lat"].iloc[0])
                slon           = float(station_row["avg_lon"].iloc[0])
                stn_cis        = float(station_row["avg_cis"].iloc[0])

                dist_km        = haversine(slat, slon, hosp["lat"], hosp["lon"])
                road_km        = dist_km * 1.4
                base_min       = road_km / AMBULANCE_SPEED_KMPH * 60
                park_delay     = stn_cis / 100 * MAX_PARKING_DELAY_MIN
                total_min      = base_min + park_delay
                survival_drop  = CARDIAC_SURVIVAL_DROP * park_delay * 100

                delay_hex = "#EF4444" if park_delay > 3 else "#F59E0B"
                result_cards = [
                    ("🛣️ Road Distance",   f"{road_km:.1f} km",   "#6C63FF"),
                    ("⏱️ Baseline Travel", f"{base_min:.1f} min",  "#6C63FF"),
                    ("🅿️ Parking Delay",   f"{park_delay:.1f} min", delay_hex),
                    ("🚑 Total Response",  f"{total_min:.1f} min", "#6C63FF"),
                ]
                rc1, rc2 = st.columns(2)
                for i, (title, val, hx) in enumerate(result_cards):
                    c = rc1 if i % 2 == 0 else rc2
                    c.markdown(f"""
<div style="background:rgba(30,33,48,0.65); border:1px solid rgba(108,99,255,0.15);
border-radius:12px; padding:14px; text-align:center; margin-bottom:8px;
backdrop-filter:blur(12px);">
  <div style="font-size:0.78rem; color:#888;">{title}</div>
  <div style="font-size:1.6rem; font-weight:800; color:{hx};">{val}</div>
</div>
""", unsafe_allow_html=True)

                # Risk badge
                if total_min < 7:
                    badge_text   = "🟢 ACCEPTABLE — Within target response time"
                    badge_bg     = "rgba(16,185,129,0.12)"
                    badge_border = "rgba(16,185,129,0.35)"
                    badge_col    = "#10B981"
                elif total_min < 10:
                    badge_text   = "🟡 ELEVATED RISK — Marginally above target"
                    badge_bg     = "rgba(245,158,11,0.12)"
                    badge_border = "rgba(245,158,11,0.35)"
                    badge_col    = "#F59E0B"
                elif total_min < 15:
                    badge_text   = "🔴 HIGH RISK — Response time critically high"
                    badge_bg     = "rgba(239,68,68,0.12)"
                    badge_border = "rgba(239,68,68,0.35)"
                    badge_col    = "#EF4444"
                else:
                    badge_text   = "🚨 CRITICAL — Life-threatening delay"
                    badge_bg     = "rgba(124,58,237,0.12)"
                    badge_border = "rgba(124,58,237,0.35)"
                    badge_col    = "#7C3AED"

                st.markdown(
                    f"<div style='background:{badge_bg}; border:1px solid {badge_border}; "
                    f"border-radius:10px; padding:10px 14px; color:{badge_col}; "
                    f"font-size:0.88rem; font-weight:600; margin:8px 0;'>{badge_text}</div>",
                    unsafe_allow_html=True,
                )

                delay_pct = park_delay / total_min if total_min > 0 else 0.0
                st.markdown(f"**Parking causes {delay_pct * 100:.0f}% of total delay**")
                st.progress(min(delay_pct, 1.0))

                if park_delay > 2:
                    st.error(
                        f"⚠️ In cardiac emergencies, this {park_delay:.1f}-minute "
                        f"parking delay reduces survival probability by ~{survival_drop:.0f}%"
                    )
        else:
            st.markdown("""
<div style="background:rgba(30,33,48,0.45); border:1px solid rgba(108,99,255,0.15);
border-radius:14px; padding:44px; text-align:center; color:#555; margin-top:8px;">
  <div style="font-size:2rem; margin-bottom:10px;">🔍</div>
  <div style="font-size:0.9rem;">Select a station and hospital,<br>then click Calculate</div>
</div>
""", unsafe_allow_html=True)


# ── Section 5: Hour-by-Hour Delay Chart ──────────────────────────────
def _render_hourly_chart(df: pd.DataFrame):
    st.markdown("### 📈 Ambulance Delay Risk Throughout the Day")

    hourly_cis   = df.groupby("hour")["cis"].mean().reindex(range(24), fill_value=0)
    hourly_delay = (hourly_cis / 100 * MAX_PARKING_DELAY_MIN).tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(24)),
        y=hourly_delay,
        fill="tozeroy",
        fillcolor="rgba(239,68,68,0.12)",
        line=dict(color="#EF4444", width=2.5),
        mode="lines+markers",
        marker=dict(size=6, color="#EF4444"),
        name="Est. Ambulance Delay",
        hovertemplate="Hour %{x}:00 — %{y:.2f} min delay<extra></extra>",
    ))

    fig.add_hline(
        y=4, line_dash="dash", line_color="#10B981", line_width=1.5,
        annotation_text="Safe threshold (4 min)",
        annotation_position="top right",
        annotation_font_color="#10B981",
    )
    fig.add_hline(
        y=6, line_dash="dash", line_color="#EF4444", line_width=1.5,
        annotation_text="Critical threshold (6 min)",
        annotation_position="top right",
        annotation_font_color="#EF4444",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
        shapes=[
            dict(type="rect", x0=7,  x1=10, y0=0, y1=8,
                 fillcolor="rgba(239,68,68,0.08)", line_width=0),
            dict(type="rect", x0=16, x1=20, y0=0, y1=8,
                 fillcolor="rgba(239,68,68,0.08)", line_width=0),
        ],
        annotations=[
            dict(x=8.5, y=7.6, text="🔴 Morning Rush", showarrow=False,
                 font=dict(color="#F97316", size=11)),
            dict(x=18.0, y=7.6, text="🔴 Evening Rush", showarrow=False,
                 font=dict(color="#F97316", size=11)),
        ],
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(0, 24, 2)),
            ticktext=[f"{h}:00" for h in range(0, 24, 2)],
            title="Hour of Day",
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(
            title="Estimated Delay (min)",
            range=[0, 8.5],
            gridcolor="rgba(255,255,255,0.05)",
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Section 6: Enforcement Impact Callout ────────────────────────────
def _render_enforcement_callout(station_agg: pd.DataFrame):
    st.markdown("### 💡 What Targeted Enforcement Can Achieve")

    top5           = station_agg.nlargest(5, "avg_cis")
    top5_avg_cis   = top5["avg_cis"].mean()
    current_delay  = top5_avg_cis / 100 * MAX_PARKING_DELAY_MIN
    after_delay    = current_delay * 0.5
    lives_saved    = int(
        (current_delay - after_delay) * CARDIAC_SURVIVAL_DROP * DAILY_AMBULANCE_CALLS * 365 / 10
    )
    annual_cardiac = int(DAILY_AMBULANCE_CALLS * 365 * CARDIAC_SURVIVAL_DROP * current_delay)
    savings_lakhs  = int(
        current_delay * 0.5 * DAILY_AMBULANCE_CALLS * 365 * 60 * VALUE_OF_TIME_RS_PER_MIN / 1e5
    )

    left, right = st.columns(2)
    with left:
        st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(239,68,68,0.15),rgba(251,146,60,0.08));
border:1px solid rgba(239,68,68,0.30); border-radius:16px; padding:24px;">
  <div style="font-size:1rem; font-weight:700; color:#EF4444; margin-bottom:16px;">
    🔴 Current State
  </div>
  <div style="color:#ccc; font-size:0.88rem; line-height:2.2;">
    <div>Avg delay in top 5 zones:
      <b style='color:#EF4444;'>{current_delay:.1f} min</b></div>
    <div>Annual cardiac risk events:
      <b style='color:#EF4444;'>~{annual_cardiac:,}</b></div>
  </div>
</div>
""", unsafe_allow_html=True)

    with right:
        st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(16,185,129,0.15),rgba(5,150,105,0.08));
border:1px solid rgba(16,185,129,0.30); border-radius:16px; padding:24px;">
  <div style="font-size:1rem; font-weight:700; color:#10B981; margin-bottom:16px;">
    ✅ After ParkWatch AI Enforcement
  </div>
  <div style="color:#ccc; font-size:0.88rem; line-height:2.2;">
    <div>Avg delay reduced to:
      <b style='color:#10B981;'>{after_delay:.1f} min</b></div>
    <div>Annual lives potentially saved:
      <b style='color:#10B981;'>~{lives_saved:,}</b></div>
    <div>Annual savings:
      <b style='color:#10B981;'>₹{savings_lakhs:,} Lakhs</b></div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align:center; color:#EF4444; font-size:1.05rem; font-weight:700; "
        f"padding:18px; background:rgba(239,68,68,0.07); border-radius:12px; "
        f"border:1px solid rgba(239,68,68,0.22);'>"
        f"🚨 Deploying ParkWatch AI at top 5 zones could save "
        f"~{lives_saved:,} lives annually in Bengaluru</div>",
        unsafe_allow_html=True,
    )


# ── Entry Point ───────────────────────────────────────────────────────
def render(df: pd.DataFrame):
    _render_hero()

    with st.spinner("Computing emergency risk metrics…"):
        station_agg = compute_station_risk(df)

    _render_city_metrics(df, station_agg)

    st.markdown(
        "<hr style='border:1px solid rgba(255,255,255,0.07); margin:28px 0;'>",
        unsafe_allow_html=True,
    )

    render_risk_map(station_agg)

    st.markdown(
        "<hr style='border:1px solid rgba(255,255,255,0.07); margin:28px 0;'>",
        unsafe_allow_html=True,
    )

    render_delay_calculator(df, station_agg)

    st.markdown(
        "<hr style='border:1px solid rgba(255,255,255,0.07); margin:28px 0;'>",
        unsafe_allow_html=True,
    )

    _render_hourly_chart(df)

    st.markdown(
        "<hr style='border:1px solid rgba(255,255,255,0.07); margin:28px 0;'>",
        unsafe_allow_html=True,
    )

    _render_enforcement_callout(station_agg)
