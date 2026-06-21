"""
tabs/tab_congestion_analytics.py — 📊 Congestion Analytics
Merged from: tab_analytics + tab_impact + tab_emergency
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from math import radians, sin, cos, sqrt, atan2
from charts.utils import hourly_counts, dow_counts, violation_type_counts, hour_dow_pivot, style_fig


# ══════════════════════════════════════════════════════════════════════════════
#  EMERGENCY — CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  EMERGENCY — HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


@st.cache_data(show_spinner=False)
def compute_station_risk(_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        _df.groupby("police_station", observed=True)
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


def _emg_render_hero():
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


def _emg_render_city_metrics(df: pd.DataFrame, station_agg: pd.DataFrame):
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


def _emg_render_risk_map(station_agg: pd.DataFrame):
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


def _emg_render_delay_calculator(df: pd.DataFrame, station_agg: pd.DataFrame):
    st.markdown("### 🧮 Response Time Calculator")
    st.markdown("<hr style='border:1px solid rgba(255,255,255,0.07);'>", unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    with left:
        stations = sorted(df["police_station"].dropna().unique())
        origin      = st.selectbox("📍 Incident Zone (Police Station)", stations, key="emg_origin")
        destination = st.selectbox(
            "🏥 Nearest Hospital", [h["name"] for h in HOSPITALS], key="emg_dest"
        )
        calc_clicked = st.button("🔍 Calculate Response Time", use_container_width=True, key="emg_calc_btn")

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
                slat       = float(station_row["avg_lat"].iloc[0])
                slon       = float(station_row["avg_lon"].iloc[0])
                stn_cis    = float(station_row["avg_cis"].iloc[0])

                dist_km    = haversine(slat, slon, hosp["lat"], hosp["lon"])
                road_km    = dist_km * 1.4
                base_min   = road_km / AMBULANCE_SPEED_KMPH * 60
                park_delay = stn_cis / 100 * MAX_PARKING_DELAY_MIN
                total_min  = base_min + park_delay
                survival_drop = CARDIAC_SURVIVAL_DROP * park_delay * 100

                delay_hex = "#EF4444" if park_delay > 3 else "#F59E0B"
                result_cards = [
                    ("🛣️ Road Distance",   f"{road_km:.1f} km",    "#6C63FF"),
                    ("⏱️ Baseline Travel", f"{base_min:.1f} min",   "#6C63FF"),
                    ("🅿️ Parking Delay",   f"{park_delay:.1f} min", delay_hex),
                    ("🚑 Total Response",  f"{total_min:.1f} min",  "#6C63FF"),
                ]
                rc1, rc2 = st.columns(2)
                for i, (title, val, hx) in enumerate(result_cards):
                    c = rc1 if i % 2 == 0 else rc2
                    c.markdown(f"""
<div style="background:rgba(30,33,48,0.65); border:1px solid rgba(108,99,255,0.15);
border-radius:12px; padding:14px; text-align:center; margin-bottom:8px;">
  <div style="font-size:0.78rem; color:#888;">{title}</div>
  <div style="font-size:1.6rem; font-weight:800; color:{hx};">{val}</div>
</div>
""", unsafe_allow_html=True)

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


def _emg_render_hourly_chart(df: pd.DataFrame):
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

    fig.add_hline(y=4, line_dash="dash", line_color="#10B981", line_width=1.5,
                  annotation_text="Safe threshold (4 min)",
                  annotation_position="top right",
                  annotation_font_color="#10B981")
    fig.add_hline(y=6, line_dash="dash", line_color="#EF4444", line_width=1.5,
                  annotation_text="Critical threshold (6 min)",
                  annotation_position="top right",
                  annotation_font_color="#EF4444")

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


def _emg_render_enforcement_callout(station_agg: pd.DataFrame):
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
    <div>Avg delay in top 5 zones: <b style='color:#EF4444;'>{current_delay:.1f} min</b></div>
    <div>Annual cardiac risk events: <b style='color:#EF4444;'>~{annual_cardiac:,}</b></div>
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
    <div>Avg delay reduced to: <b style='color:#10B981;'>{after_delay:.1f} min</b></div>
    <div>Annual lives potentially saved: <b style='color:#10B981;'>~{lives_saved:,}</b></div>
    <div>Annual savings: <b style='color:#10B981;'>₹{savings_lakhs:,} Lakhs</b></div>
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


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION RENDERS
# ══════════════════════════════════════════════════════════════════════════════

def _render_analytics(df):
    st.markdown(
        "<p class='section-header'>Deep-Dive Analytics</p>"
        "<p class='section-sub'>Temporal patterns and violation type breakdown</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='pw-info-banner'>"
        "ℹ️ Charts update automatically based on your uploaded data. "
        "Hover over any chart for exact values — click legend items to show/hide series."
        "</div>",
        unsafe_allow_html=True,
    )

    a1, a2 = st.columns(2)
    with a1:
        hc = hourly_counts(df)
        rush_hours = set(range(7, 11)) | set(range(16, 21))
        bar_clr = ["#FF4B4B" if h in rush_hours else "#6C63FF" for h in range(24)]
        fig_hour = go.Figure(
            go.Bar(
                x=list(range(24)),
                y=hc.values,
                marker=dict(color=bar_clr, line=dict(width=0)),
                text=hc.values,
                textposition="outside",
                textfont=dict(size=9, color="#888"),
            )
        )
        fig_hour.update_layout(
            title=dict(text="Violations by Hour of Day", font=dict(size=14, color="#ddd")),
            xaxis=dict(title="Hour", dtick=1),
            yaxis_title="Count",
            annotations=[
                dict(x=8.5, y=hc.max() * 1.1, text="🔴 Rush Hours",
                     showarrow=False, font=dict(color="#FF4B4B", size=10))
            ],
        )
        style_fig(fig_hour, 400)
        st.plotly_chart(fig_hour, use_container_width=True)

    with a2:
        dc = dow_counts(df)
        fig_dow = go.Figure(
            go.Bar(
                x=dc.index,
                y=dc.values,
                marker=dict(color=["#6C63FF"] * 5 + ["#00D2FF"] * 2, line=dict(width=0)),
                text=dc.values,
                textposition="outside",
                textfont=dict(size=10, color="#888"),
            )
        )
        fig_dow.update_layout(
            title=dict(text="Violations by Day of Week", font=dict(size=14, color="#ddd")),
            xaxis_title=None,
            yaxis_title="Count",
        )
        style_fig(fig_dow, 400)
        st.plotly_chart(fig_dow, use_container_width=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    with b1:
        vtc = violation_type_counts(df)
        n_vt = len(vtc)
        vt_colors = [
            f"rgba({108 + int(i * 100 / max(n_vt, 1))}, {99 + int(i * 60 / max(n_vt, 1))}, 255, 0.85)"
            for i in range(n_vt)
        ][::-1]
        fig_vt = go.Figure(
            go.Bar(
                x=vtc.values[::-1],
                y=vtc.index[::-1],
                orientation="h",
                marker=dict(color=vt_colors, line=dict(width=0)),
                text=vtc.values[::-1],
                textposition="outside",
                textfont=dict(size=10, color="#aaa"),
            )
        )
        fig_vt.update_layout(
            title=dict(text="Violation Type Breakdown", font=dict(size=14, color="#ddd")),
            xaxis_title="Count",
            yaxis_title=None,
        )
        style_fig(fig_vt, 450)
        st.plotly_chart(fig_vt, use_container_width=True)

    with b2:
        pivot = hour_dow_pivot(df)
        fig_hm = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=[f"{h:02d}:00" for h in pivot.index],
                colorscale=[
                    [0, "#0E1117"],
                    [0.25, "#2a1a5e"],
                    [0.5, "#6C63FF"],
                    [0.75, "#00D2FF"],
                    [1.0, "#FF4B4B"],
                ],
                hovertemplate="Day: %{x}<br>Hour: %{y}<br>Violations: %{z}<extra></extra>",
                colorbar=dict(title="Count", tickfont=dict(color="#888")),
            )
        )
        fig_hm.update_layout(
            title=dict(text="Hour × Day Heatmap", font=dict(size=14, color="#ddd")),
            xaxis_title=None,
            yaxis=dict(title=None, autorange="reversed"),
        )
        style_fig(fig_hm, 450)
        st.plotly_chart(fig_hm, use_container_width=True)


def _render_impact(df):
    st.markdown(
        "<p class='section-header'>Economic Impact Calculator</p>"
        "<p class='section-sub'>Quantify the economic cost of parking-induced congestion</p>",
        unsafe_allow_html=True,
    )

    inp1, inp2 = st.columns(2)
    with inp1:
        reduction_pct = st.slider(
            "🔽 Enforcement Reduction %", min_value=10, max_value=50, value=30, step=5,
            help="Estimated % reduction in violations through targeted enforcement",
        )
    with inp2:
        time_value = st.number_input(
            "⏱️ Value of Time (₹/hour)", min_value=50, max_value=1000, value=200, step=50,
            help="Average economic value of one person-hour lost in congestion",
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    date_range_days = (df["created_datetime"].max() - df["created_datetime"].min()).days
    date_range_days = max(date_range_days, 1)
    avg_daily = len(df) / date_range_days
    delay_per_violation = 0.5
    daily_cost = avg_daily * delay_per_violation * time_value
    annual_cost = daily_cost * 365
    annual_savings = annual_cost * reduction_pct / 100

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""<div class='kpi-card'>
                <div class='kpi-label'>Avg Daily Violations</div>
                <div class='kpi-value'>{avg_daily:,.0f}</div>
                <div class='kpi-sub'>Over {date_range_days} days</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""<div class='impact-card'>
                <div class='kpi-label'>Daily Congestion Cost</div>
                <div class='kpi-value'>₹{daily_cost:,.0f}</div>
                <div class='kpi-sub'>Est. delay cost</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""<div class='impact-card'>
                <div class='kpi-label'>Annual Cost</div>
                <div class='kpi-value'>₹{annual_cost / 1e7:,.1f} Cr</div>
                <div class='kpi-sub'>Projected yearly</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""<div class='savings-card'>
                <div class='kpi-label'>Potential Savings</div>
                <div class='kpi-value'>₹{annual_savings / 1e7:,.1f} Cr</div>
                <div class='kpi-sub'>At {reduction_pct}% reduction</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    station_agg = (
        df.groupby("police_station", observed=True)
        .agg(violation_count=("id", "size"), avg_severity=("violation_severity", "mean"))
        .reset_index()
    )
    station_agg["est_daily"] = station_agg["violation_count"] / date_range_days
    station_agg["daily_cost"] = station_agg["est_daily"] * delay_per_violation * time_value
    station_agg = station_agg.sort_values("daily_cost", ascending=False).head(15)

    n_sc = len(station_agg)
    cost_colors = [
        f"rgba({int(255 - i * 8)}, {int(75 + i * 12)}, {int(75 + i * 6)}, 0.9)"
        for i in range(n_sc)
    ]
    fig_cost = go.Figure(
        go.Bar(
            x=station_agg["daily_cost"].values[::-1],
            y=station_agg["police_station"].values[::-1],
            orientation="h",
            marker=dict(color=cost_colors[::-1], line=dict(width=0)),
            text=[f"₹{v:,.0f}" for v in station_agg["daily_cost"].values[::-1]],
            textposition="outside",
            textfont=dict(size=10, color="#ddd"),
        )
    )
    fig_cost.update_layout(
        title=dict(text="Estimated Daily Congestion Cost — Top 15 Stations",
                   font=dict(size=15, color="#ddd")),
        xaxis_title="Daily Cost (₹)",
        yaxis_title=None,
    )
    style_fig(fig_cost, 480)
    st.plotly_chart(fig_cost, use_container_width=True)

    st.markdown("**📋 Assumptions & Methodology**")
    st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Delay per violation | {delay_per_violation} person-hours |
| Value of time | ₹{time_value}/hour |
| Data period | {date_range_days} days |
| Enforcement reduction | {reduction_pct}% |

**Model**: Each parking violation is estimated to cause **{delay_per_violation} person-hours**
of cumulative congestion delay (accounting for cascading effects on traffic flow).
The economic cost = violations × delay × value-of-time.
Savings assume targeted enforcement at high-EPI stations reduces violations by the selected percentage.
    """)


def _render_emergency(df: pd.DataFrame):
    _emg_render_hero()

    with st.spinner("Computing emergency risk metrics…"):
        station_agg = compute_station_risk(df)

    _emg_render_city_metrics(df, station_agg)

    st.markdown("<hr style='border:1px solid rgba(255,255,255,0.07); margin:28px 0;'>",
                unsafe_allow_html=True)
    _emg_render_risk_map(station_agg)

    st.markdown("<hr style='border:1px solid rgba(255,255,255,0.07); margin:28px 0;'>",
                unsafe_allow_html=True)
    _emg_render_delay_calculator(df, station_agg)

    st.markdown("<hr style='border:1px solid rgba(255,255,255,0.07); margin:28px 0;'>",
                unsafe_allow_html=True)
    _emg_render_hourly_chart(df)

    st.markdown("<hr style='border:1px solid rgba(255,255,255,0.07); margin:28px 0;'>",
                unsafe_allow_html=True)
    _emg_render_enforcement_callout(station_agg)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS FOR EXPANDER SECTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _render_temporal_charts(df):
    """Weekday vs weekend hourly CIS curves + hourly/DOW bars + violation breakdown."""
    wkday = df[df["is_weekend"] == 0].groupby("hour")["cis"].mean().reindex(range(24), fill_value=0)
    wkend = df[df["is_weekend"] == 1].groupby("hour")["cis"].mean().reindex(range(24), fill_value=0)
    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(
        x=list(range(24)), y=wkday.values, name="Weekday",
        fill="tozeroy", fillcolor="rgba(108,99,255,0.12)",
        line=dict(color="#6C63FF", width=2.5),
        hovertemplate="Hour %{x}:00 — Avg CIS %{y:.1f}<extra>Weekday</extra>",
    ))
    fig_wave.add_trace(go.Scatter(
        x=list(range(24)), y=wkend.values, name="Weekend",
        fill="tozeroy", fillcolor="rgba(0,210,255,0.10)",
        line=dict(color="#00D2FF", width=2.5),
        hovertemplate="Hour %{x}:00 — Avg CIS %{y:.1f}<extra>Weekend</extra>",
    ))
    fig_wave.update_layout(
        title=dict(text="Avg CIS by Hour — Weekday vs Weekend", font=dict(size=14, color="#ddd")),
        xaxis=dict(title="Hour of Day", dtick=2),
        yaxis_title="Avg CIS",
    )
    style_fig(fig_wave, 360)
    st.plotly_chart(fig_wave, use_container_width=True)

    a1, a2 = st.columns(2)
    with a1:
        hc = hourly_counts(df)
        rush_hours = set(range(7, 11)) | set(range(16, 21))
        bar_clr = ["#FF4B4B" if h in rush_hours else "#6C63FF" for h in range(24)]
        fig_hour = go.Figure(go.Bar(
            x=list(range(24)), y=hc.values,
            marker=dict(color=bar_clr, line=dict(width=0)),
            text=hc.values, textposition="outside", textfont=dict(size=9, color="#888"),
        ))
        fig_hour.update_layout(
            title=dict(text="Violations by Hour of Day", font=dict(size=14, color="#ddd")),
            xaxis=dict(title="Hour", dtick=1), yaxis_title="Count",
            annotations=[dict(x=8.5, y=hc.max() * 1.1, text="🔴 Rush Hours",
                showarrow=False, font=dict(color="#FF4B4B", size=10))],
        )
        style_fig(fig_hour, 360)
        st.plotly_chart(fig_hour, use_container_width=True)
    with a2:
        dc = dow_counts(df)
        fig_dow = go.Figure(go.Bar(
            x=dc.index, y=dc.values,
            marker=dict(color=["#6C63FF"] * 5 + ["#00D2FF"] * 2, line=dict(width=0)),
            text=dc.values, textposition="outside", textfont=dict(size=10, color="#888"),
        ))
        fig_dow.update_layout(
            title=dict(text="Violations by Day of Week", font=dict(size=14, color="#ddd")),
            xaxis_title=None, yaxis_title="Count",
        )
        style_fig(fig_dow, 360)
        st.plotly_chart(fig_dow, use_container_width=True)

    vtc = violation_type_counts(df)
    n_vt = len(vtc)
    vt_colors = [
        f"rgba({108 + int(i * 100 / max(n_vt, 1))}, {99 + int(i * 60 / max(n_vt, 1))}, 255, 0.85)"
        for i in range(n_vt)
    ][::-1]
    fig_vt = go.Figure(go.Bar(
        x=vtc.values[::-1], y=vtc.index[::-1], orientation="h",
        marker=dict(color=vt_colors, line=dict(width=0)),
        text=vtc.values[::-1], textposition="outside", textfont=dict(size=10, color="#aaa"),
    ))
    fig_vt.update_layout(
        title=dict(text="Violation Type Breakdown", font=dict(size=14, color="#ddd")),
        xaxis_title="Count", yaxis_title=None,
    )
    style_fig(fig_vt, 400)
    st.plotly_chart(fig_vt, use_container_width=True)


def _render_cis_histogram(df):
    fig = px.histogram(
        df, x="cis", nbins=50,
        color_discrete_sequence=["#6C63FF"],
        labels={"cis": "CIS Score"},
        title="Distribution of Congestion Impact Scores",
    )
    fig.update_traces(marker_line_width=0, opacity=0.85)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=380, bargap=0.02,
        xaxis_title="CIS Score", yaxis_title="Count",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_vehicle_hour_chart(df):
    if "vehicle_type" not in df.columns:
        st.info("No vehicle_type column in data.")
        return
    top_types = df["vehicle_type"].value_counts().head(6).index.tolist()
    veh_hour = (
        df[df["vehicle_type"].isin(top_types)]
        .groupby(["hour", "vehicle_type"], observed=True)
        .size().reset_index(name="count")
    )
    palette = ["#6C63FF", "#00D2FF", "#7C3AED", "#3B82F6", "#10B981", "#F59E0B"]
    fig = go.Figure()
    for i, vtype in enumerate(top_types):
        sub = veh_hour[veh_hour["vehicle_type"] == vtype]
        fig.add_trace(go.Bar(
            x=sub["hour"], y=sub["count"], name=vtype,
            marker_color=palette[i % len(palette)],
        ))
    fig.update_layout(
        barmode="stack",
        title=dict(text="Violations by Vehicle Type × Hour", font=dict(size=14, color="#ddd")),
        xaxis=dict(title="Hour of Day", dtick=2),
        yaxis_title="Violation Count",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        legend=dict(orientation="h", y=-0.22),
        margin=dict(l=0, r=0, t=40, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_anomaly_detail(df):
    stn_stats = (
        df.groupby("police_station", observed=True)["cis"]
        .agg(avg_cis="mean", violation_count="count")
        .reset_index()
    )
    mu = stn_stats["avg_cis"].mean()
    sigma = stn_stats["avg_cis"].std()
    stn_stats["z_score"] = (stn_stats["avg_cis"] - mu) / sigma
    anomalies = stn_stats[stn_stats["z_score"] > 2.0].sort_values("z_score", ascending=False)

    if anomalies.empty:
        st.info("No anomalous zones detected at z-score > 2.0 threshold.")
        return

    st.markdown(f"**{len(anomalies)} zones significantly above city-average CIS (z > 2.0)**")
    for _, row in anomalies.iterrows():
        badge = "🔴" if row["z_score"] > 3 else "🟠"
        st.markdown(
            f"{badge} **{row['police_station']}** — "
            f"Avg CIS: `{row['avg_cis']:.1f}` &nbsp;·&nbsp; "
            f"z-score: `{row['z_score']:.2f}` &nbsp;·&nbsp; "
            f"Violations: `{int(row['violation_count']):,}`"
        )

    st.divider()

    numeric_cols = ["violation_severity", "vehicle_size_score", "time_factor", "junction_factor"]
    available = [c for c in numeric_cols if c in df.columns]
    if available:
        st.markdown("**Feature correlation with CIS (proxy for model importance)**")
        corrs = df[available + ["cis"]].corr()["cis"].drop("cis").abs().sort_values(ascending=False)
        fig_imp = go.Figure(go.Bar(
            x=corrs.values, y=corrs.index, orientation="h",
            marker=dict(
                color=[f"rgba(108,99,255,{0.45 + 0.55 * v:.2f})" for v in corrs.values],
                line=dict(width=0),
            ),
            text=[f"{v:.3f}" for v in corrs.values],
            textposition="outside",
            textfont=dict(size=10, color="#aaa"),
        ))
        fig_imp.update_layout(
            title=dict(text="CIS Feature Correlation", font=dict(size=13, color="#ddd")),
            xaxis_title="|Correlation with CIS|", yaxis_title=None,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=260,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    peak_risk_hour = int(df.groupby("hour")["cis"].mean().idxmax())
    st.markdown(
        f"**Model prediction:** Peak high-risk hour is `{peak_risk_hour:02d}:00` "
        f"(avg CIS = `{df[df['hour'] == peak_risk_hour]['cis'].mean():.1f}`)"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def render(df):
    # ── Row 1: Three impact KPIs ──────────────────────────────────────────────
    date_range_days = max(
        (df["created_datetime"].max() - df["created_datetime"].min()).days, 1
    )
    avg_daily = len(df) / date_range_days
    monthly_cost_cr = avg_daily * 0.5 * 200 * 30 / 1e7
    avg_delay_min = df["cis"].mean() / 100 * MAX_PARKING_DELAY_MIN
    productivity_hrs_month = avg_daily * 0.5 * 30

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Monthly congestion cost", f"₹{monthly_cost_cr:.1f} Cr")
    with col2:
        st.metric("Avg ambulance delay", f"+{avg_delay_min:.1f} min")
    with col3:
        st.metric("Productivity lost/month", f"{productivity_hrs_month:,.0f} hrs")

    # ── Row 2: day × hour heatmap ─────────────────────────────────────────────
    st.subheader("Violation intensity — day × hour")
    pivot = hour_dow_pivot(df)
    fig_hm = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=[f"{h:02d}:00" for h in pivot.index],
        colorscale=[
            [0,    "#0E1117"],
            [0.25, "#2a1a5e"],
            [0.5,  "#6C63FF"],
            [0.75, "#00D2FF"],
            [1.0,  "#FF4B4B"],
        ],
        hovertemplate="Day: %{x}<br>Hour: %{y}<br>Violations: %{z}<extra></extra>",
        colorbar=dict(title="Count", tickfont=dict(color="#888")),
    ))
    fig_hm.update_layout(
        title=dict(text="Hour × Day Heatmap", font=dict(size=14, color="#ddd")),
        xaxis_title=None,
        yaxis=dict(title=None, autorange="reversed"),
    )
    style_fig(fig_hm, 420)
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── Anomaly count (always visible) ───────────────────────────────────────
    stn_cis = df.groupby("police_station", observed=True)["cis"].mean()
    anomaly_count = int((stn_cis > stn_cis.mean() + 2.0 * stn_cis.std()).sum())
    st.info(
        f"🔍 {anomaly_count} anomalous zones detected by AI model — expand below for details"
    )

    st.divider()

    # ── Secondary sections in expanders ──────────────────────────────────────
    with st.expander("📈 Weekday vs weekend congestion curves", expanded=False):
        _render_temporal_charts(df)

    with st.expander("🚑 Ambulance delay breakdown by zone", expanded=False):
        with st.spinner("Computing emergency risk metrics…"):
            station_agg = compute_station_risk(df)
        _emg_render_risk_map(station_agg)
        _emg_render_delay_calculator(df, station_agg)
        _emg_render_hourly_chart(df)

    with st.expander("💰 Economic cost breakdown by zone", expanded=False):
        _render_impact(df)

    with st.expander("📊 CIS score distribution", expanded=False):
        _render_cis_histogram(df)

    with st.expander("🚗 Vehicle type × hour breakdown", expanded=False):
        _render_vehicle_hour_chart(df)

    with st.expander("🤖 AI model predictions & anomaly detail", expanded=False):
        _render_anomaly_detail(df)
