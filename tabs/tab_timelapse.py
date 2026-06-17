"""
tabs/tab_timelapse.py — ⏳ Time-Lapse Tab
Pre-aggregates all 24 × 3 hourly buckets once; slider is pure lookup.
Animation plays entirely in Plotly (browser-side) — no st.rerun() loop,
no WebSocket churn.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Helper ────────────────────────────────────────────────────────────
def format_hour(h: int) -> str:
    if h == 0:  return "12:00 AM (Midnight)"
    if h < 12:  return f"{h}:00 AM"
    if h == 12: return "12:00 PM (Noon)"
    return f"{h - 12}:00 PM"


_INSIGHTS = [
    (range(0, 5),   "🌙", "#3B82F6", "Low Activity Period",
     "Minimal violations. Night security and early morning patrols can stand down. Save resources for rush hours."),
    (range(5, 7),   "🌅", "#F59E0B", "Early Morning Build-Up",
     "Violations starting to increase. Market and delivery vehicles begin parking. Pre-position tow trucks near commercial areas."),
    (range(7, 11),  "🔴", "#EF4444", "MORNING RUSH PEAK",
     "Maximum violation density. Deploy all available patrol teams immediately. Focus on junctions and main roads."),
    (range(11, 16), "🟡", "#F59E0B", "Midday Activity",
     "Moderate violations, mainly commercial parking. Target market areas and shopping zones."),
    (range(16, 21), "🔴", "#EF4444", "EVENING RUSH PEAK",
     "Second peak period. Tech park exits and commercial areas critical. All teams on active patrol."),
    (range(21, 24), "🌆", "#6C63FF", "Evening Wind-Down",
     "Violations reducing. Maintain presence near restaurants and entertainment zones."),
]

_HEAT_SCALE = [
    [0.00, "#1E3A5F"],
    [0.30, "#1565C0"],
    [0.50, "#F57F17"],
    [0.75, "#E53935"],
    [1.00, "#7B1FA2"],
]


# ── Pre-computation ───────────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Precomputing time-lapse data — runs once…")
def precompute_hourly_data(_df: pd.DataFrame) -> dict:
    """
    Returns nested dict:
      result[day_filter][hour] = {points, count, avg_cis, top_station, top_station_count}
      result["daily_avg"]      = float
      result["city_avg_cis"]   = float
    """
    def _build(subset: pd.DataFrame) -> dict:
        hourly: dict = {}
        for h in range(24):
            hdf = subset[subset["hour"] == h]
            n   = len(hdf)
            if n > 0:
                pts     = (
                    hdf[["latitude", "longitude", "cis"]]
                    .dropna()
                    .sample(min(3000, n), random_state=42)
                )
                vc      = hdf["police_station"].value_counts()
                top_stn = vc.idxmax()
                top_cnt = int(vc.iloc[0])
                avg_cis = float(hdf["cis"].mean())
            else:
                pts     = pd.DataFrame(columns=["latitude", "longitude", "cis"])
                top_stn = "N/A"
                top_cnt = 0
                avg_cis = 0.0
            hourly[h] = {
                "points":            pts,
                "count":             n,
                "avg_cis":           avg_cis,
                "top_station":       top_stn,
                "top_station_count": top_cnt,
            }
        return hourly

    return {
        "All Days":     _build(_df),
        "Weekdays":     _build(_df[_df["is_weekend"] == 0]),
        "Weekends":     _build(_df[_df["is_weekend"] == 1]),
        "daily_avg":    len(_df) / max(_df["created_datetime"].dt.date.nunique(), 1),
        "city_avg_cis": float(_df["cis"].mean()),
    }


# ── Section 1: Hero ───────────────────────────────────────────────────
def _render_hero():
    st.markdown("""
<div style="background:linear-gradient(135deg,rgba(108,99,255,0.12),rgba(0,210,255,0.08));
border:1px solid rgba(108,99,255,0.25); border-radius:20px; padding:28px 32px;
text-align:center; margin-bottom:20px;">
  <div style="font-size:2.8rem; margin-bottom:6px;">⏳</div>
  <h1 style="background:linear-gradient(90deg,#6C63FF,#00D2FF);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  font-size:1.9rem; font-weight:800; margin:0;">
    Live Violation Time-Lapse
  </h1>
  <p style="color:#aaa; font-size:1rem; margin-top:8px;">
    Watch parking violations surge during rush hour and fade at night —
    across all of Bengaluru
  </p>
</div>
""", unsafe_allow_html=True)


# ── Section 2: Controls (slider + day filter only) ────────────────────
def render_controls():
    left, center = st.columns([1, 3])

    with left:
        day_filter = st.radio(
            "📅 Day Type",
            ["All Days", "Weekdays", "Weekends"],
            key="tl_day_filter",
        )

    with center:
        selected_hour = st.slider(
            "🕐 Explore Hour",
            min_value=0, max_value=23,
            key="hour_slider",
            format="%d",
        )
        st.markdown(
            f"<h3 style='text-align:center; color:#6C63FF; margin:6px 0 0 0;'>"
            f"⏰ {format_hour(selected_hour)}</h3>",
            unsafe_allow_html=True,
        )

    return selected_hour, day_filter


# ── Section 3: Hour Strip ─────────────────────────────────────────────
def _render_hour_strip(selected_hour: int):
    boxes = ""
    for h in range(24):
        if h == selected_hour:
            bg     = "background:linear-gradient(135deg,#6C63FF,#00D2FF);"
            border = "border:2px solid #fff;"
            txt    = "color:white; font-weight:800;"
        elif (7 <= h <= 10) or (16 <= h <= 20):
            bg     = "background:rgba(239,68,68,0.4);"
            border = "border:1px solid rgba(239,68,68,0.3);"
            txt    = "color:#fca5a5;"
        else:
            bg     = "background:rgba(30,33,48,0.6);"
            border = "border:1px solid rgba(255,255,255,0.05);"
            txt    = "color:#555;"
        boxes += (
            f"<div style='{bg}{border}{txt}"
            f"border-radius:6px; width:3.8%; display:inline-block;"
            f"text-align:center; padding:4px 0; font-size:0.7rem; margin:1px;'>{h}</div>"
        )
    st.markdown(f"""
<div style="margin:12px 0 20px 0;">
  <div style="font-size:0.75rem; color:#888; margin-bottom:6px;">
    🔴 Rush Hour &nbsp;&nbsp; 🟣 Selected &nbsp;&nbsp; ⬛ Off-Peak
  </div>
  <div style="width:100%;">{boxes}</div>
</div>
""", unsafe_allow_html=True)


# ── Section 4: Live Stats ─────────────────────────────────────────────
def _render_live_stats(current: dict, daily_avg: float, city_avg_cis: float):
    hourly_avg = daily_avg / 24
    cis        = current["avg_cis"]
    cis_delta  = cis - city_avg_cis

    if cis > 60:   intensity = "🔴 CRITICAL"
    elif cis > 40: intensity = "🟠 HIGH"
    elif cis > 20: intensity = "🟡 MODERATE"
    else:          intensity = "🟢 LOW"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "🚨 Violations This Hour",
        f"{current['count']:,}",
        delta=f"{current['count'] - hourly_avg:+.0f} vs hourly avg",
    )
    c2.metric(
        "🔥 Avg CIS Score",
        f"{cis:.1f}",
        delta=f"{cis_delta:+.1f} vs city avg",
    )
    c3.metric(
        "📍 Top Hotspot",
        current["top_station"],
        delta=f"{current['top_station_count']} violations here",
        delta_color="off",
    )
    c4.metric(
        "⚡ Intensity",
        intensity,
        delta=f"CIS {cis:.1f} / 100",
        delta_color="off",
    )


# ── Section 5: Static Density Map (selected hour) ─────────────────────
def render_map(current: dict, selected_hour: int):
    points = current["points"]

    if len(points) == 0:
        st.info(f"No violation data available for {format_hour(selected_hour)}.")
        return

    fig = px.density_mapbox(
        points,
        lat="latitude",
        lon="longitude",
        z="cis",
        radius=18,
        center={"lat": 12.9716, "lon": 77.5946},
        zoom=11,
        mapbox_style="carto-darkmatter",
        color_continuous_scale=_HEAT_SCALE,
        range_color=[0, 100],
        opacity=0.85,
        title=f"Violation Density — {format_hour(selected_hour)}",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=480,
        coloraxis_colorbar=dict(
            title="CIS Score",
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["Minimal", "Low", "Moderate", "High", "Critical"],
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.1)",
            tickfont=dict(color="#bbb"),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    if (7 <= selected_hour <= 10) or (16 <= selected_hour <= 20):
        fig.add_annotation(
            text="⚠️ RUSH HOUR — Maximum Enforcement Needed",
            xref="paper", yref="paper",
            x=0.5, y=0.97,
            showarrow=False,
            font=dict(size=14, color="#EF4444"),
            bgcolor="rgba(239,68,68,0.15)",
            bordercolor="#EF4444",
            borderwidth=1,
            borderpad=8,
        )

    st.plotly_chart(fig, use_container_width=True, key=f"tl_map_{selected_hour}")


# ── Section 6: Plotly Browser-Side Animation (no st.rerun) ───────────
def render_animation(hourly_data: dict):
    """
    All 24 frames pre-loaded into a single Plotly figure.
    Playback is driven by Plotly's own JS engine in the browser —
    zero Python reruns, zero WebSocket reconnections.
    """
    st.markdown("### 🎬 24-Hour Auto-Animation")
    st.markdown(
        "<p style='color:#888; font-size:0.88rem; margin-top:-8px; margin-bottom:12px;'>"
        "Click ▶ to watch violations build up and fade away — "
        "runs entirely in your browser with no page reloads</p>",
        unsafe_allow_html=True,
    )

    # Combine per-hour samples (800 pts/frame keeps payload manageable)
    frames: list[pd.DataFrame] = []
    for h in range(24):
        pts = hourly_data[h]["points"]
        if len(pts) > 0:
            sample = pts.sample(min(800, len(pts)), random_state=42).copy()
            sample["hour"] = h
            frames.append(sample)

    if not frames:
        st.info("No data available for animation.")
        return

    anim_df = pd.concat(frames, ignore_index=True)

    fig = px.density_mapbox(
        anim_df,
        lat="latitude",
        lon="longitude",
        z="cis",
        animation_frame="hour",
        radius=16,
        center={"lat": 12.9716, "lon": 77.5946},
        zoom=11,
        mapbox_style="carto-darkmatter",
        color_continuous_scale=_HEAT_SCALE,
        range_color=[0, 100],
        opacity=0.85,
        title="Violation Density — 24-Hour Time-Lapse",
    )

    # Tune playback speed and relabel the slider steps
    try:
        play_args = fig.layout.updatemenus[0].buttons[0].args[1]
        play_args["frame"]["duration"]      = 900
        play_args["transition"]["duration"] = 200
        for i, step in enumerate(fig.layout.sliders[0]["steps"]):
            step["label"] = format_hour(i)
    except (IndexError, KeyError, TypeError):
        pass  # layout not yet populated — Plotly defaults are fine

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=520,
        coloraxis_colorbar=dict(
            title="CIS Score",
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["Minimal", "Low", "Moderate", "High", "Critical"],
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.1)",
            tickfont=dict(color="#bbb"),
        ),
        margin=dict(l=0, r=0, t=40, b=60),
    )

    st.plotly_chart(fig, use_container_width=True, key="tl_animation")


# ── Section 7: 24-Hour Comparison Chart ──────────────────────────────
def render_comparison_chart(hourly_data: dict, selected_hour: int):
    st.markdown("### 📊 Violation Intensity — Full 24 Hours")

    stats = pd.DataFrame({
        "hour":    list(range(24)),
        "count":   [hourly_data[h]["count"]   for h in range(24)],
        "avg_cis": [hourly_data[h]["avg_cis"] for h in range(24)],
    })

    bar_colors = [
        "#6C63FF" if h == selected_hour
        else "#EF4444" if (7 <= h <= 10 or 16 <= h <= 20)
        else "#3B4A6B"
        for h in range(24)
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=stats["hour"],
        y=stats["count"],
        marker_color=bar_colors,
        name="Violations",
        hovertemplate="Hour %{x}:00<br>Count: %{y:,}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=stats["hour"],
        y=stats["avg_cis"],
        mode="lines+markers",
        line=dict(color="#F59E0B", width=2),
        marker=dict(size=4),
        name="Avg CIS",
        yaxis="y2",
        hovertemplate="Hour %{x}:00<br>Avg CIS: %{y:.1f}<extra></extra>",
    ))

    fig.add_vline(
        x=selected_hour,
        line_dash="dash", line_color="#6C63FF", line_width=2,
    )

    sel_count = int(stats.loc[stats["hour"] == selected_hour, "count"].iloc[0])
    fig.add_annotation(
        x=selected_hour, y=sel_count,
        text="▲ You are here",
        showarrow=True, arrowhead=2,
        arrowcolor="#6C63FF",
        ax=0, ay=-32,
        font=dict(color="#6C63FF", size=10),
        bgcolor="rgba(30,33,48,0.85)",
        borderpad=4,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        bargap=0.15,
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(0, 24, 3)),
            ticktext=[format_hour(h) for h in range(0, 24, 3)],
            title="Hour of Day",
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(
            title="Violations",
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis2=dict(
            title="Avg CIS",
            overlaying="y", side="right",
            showgrid=False,
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"tl_bar_{selected_hour}")


# ── Section 8: Insight Callout ────────────────────────────────────────
def _render_insight(selected_hour: int):
    for hour_range, emoji, color, title, msg in _INSIGHTS:
        if selected_hour in hour_range:
            st.markdown(f"""
<div style="background:rgba(30,33,48,0.7); border-left:4px solid {color};
border-radius:10px; padding:16px 20px; margin-top:16px;">
  <div style="font-size:1.1rem; font-weight:700; color:{color};">
    {emoji} {title}
  </div>
  <div style="color:#bbb; margin-top:6px; font-size:0.92rem;">{msg}</div>
</div>
""", unsafe_allow_html=True)
            break


# ── Entry Point ───────────────────────────────────────────────────────
def render(df: pd.DataFrame):
    with st.spinner("⏳ Precomputing hourly data…"):
        all_hourly   = precompute_hourly_data(df)
    daily_avg        = all_hourly["daily_avg"]
    city_avg_cis     = all_hourly["city_avg_cis"]

    if "hour_slider" not in st.session_state:
        st.session_state["hour_slider"] = 9

    _render_hero()
    selected_hour, day_filter = render_controls()
    _render_hour_strip(selected_hour)

    hourly_data = all_hourly[day_filter]
    current     = hourly_data[selected_hour]

    st.markdown(
        "<hr style='border:1px solid rgba(255,255,255,0.07); margin:16px 0;'>",
        unsafe_allow_html=True,
    )
    _render_live_stats(current, daily_avg, city_avg_cis)

    st.markdown(
        "<hr style='border:1px solid rgba(255,255,255,0.07); margin:16px 0;'>",
        unsafe_allow_html=True,
    )

    map_tab, anim_tab = st.tabs(["🗺️ Selected Hour", "🎬 24-Hour Animation"])
    with map_tab:
        render_map(current, selected_hour)
    with anim_tab:
        render_animation(hourly_data)

    st.markdown(
        "<hr style='border:1px solid rgba(255,255,255,0.07); margin:16px 0;'>",
        unsafe_allow_html=True,
    )
    render_comparison_chart(hourly_data, selected_hour)
    _render_insight(selected_hour)
