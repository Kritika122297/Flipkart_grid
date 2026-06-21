"""
tabs/tab_command_center.py — 🏠 Command Center
Merged from: tab_overview + tab_heatmap + tab_timelapse
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from charts.utils import station_counts, vehicle_counts, ACCENT_SEQ, style_fig


# ══════════════════════════════════════════════════════════════════════════════
#  HEATMAP — CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_BGLR_CENTER = [12.9716, 77.5946]
_MAX_HEAT_PTS = 25_000
_MAX_SCATTER_PTS = 2_000
_TOP_ENF_N = 10
_TOP_HOTSPOT_N = 10
_TOP_VIOL_N = 6
_HIGH_IMPACT_CIS = 50.0

_HEAT_GRADIENT = {
    0.2: "#38BDF8",
    0.4: "#34D399",
    0.6: "#FBBF24",
    0.8: "#F87171",
    1.0: "#A78BFA",
}

_LANDMARKS = [
    {"name": "Silk Board Junction",  "lat": 12.9174, "lon": 77.6229, "icon": "\U0001f534"},
    {"name": "KR Puram Bridge",      "lat": 13.0081, "lon": 77.6933, "icon": "\U0001f534"},
    {"name": "Marathahalli Bridge",  "lat": 12.9558, "lon": 77.7016, "icon": "\U0001f534"},
    {"name": "Hebbal Flyover",       "lat": 13.0450, "lon": 77.5960, "icon": "\U0001f534"},
    {"name": "Koramangala",          "lat": 12.9352, "lon": 77.6245, "icon": "\U0001f4cd"},
    {"name": "Indiranagar",          "lat": 12.9784, "lon": 77.6408, "icon": "\U0001f4cd"},
    {"name": "MG Road",              "lat": 12.9758, "lon": 77.6094, "icon": "\U0001f4cd"},
    {"name": "HSR Layout",           "lat": 12.9116, "lon": 77.6389, "icon": "\U0001f4cd"},
    {"name": "Whitefield",           "lat": 12.9698, "lon": 77.7499, "icon": "\U0001f4cd"},
    {"name": "Electronic City",      "lat": 12.8456, "lon": 77.6603, "icon": "\U0001f4cd"},
    {"name": "Outer Ring Road",      "lat": 12.9270, "lon": 77.6782, "icon": "\U0001f6e3️"},
    {"name": "Bellandur",            "lat": 12.9262, "lon": 77.6767, "icon": "\U0001f534"},
    {"name": "Banashankari",         "lat": 12.9255, "lon": 77.5468, "icon": "\U0001f4cd"},
]

_MAP_TILES = {
    "Dark":   "CartoDB dark_matter",
    "Light":  "CartoDB positron",
    "Street": "OpenStreetMap",
}

_POPUP_STYLE = {
    "Dark":   ("#1a1a2e", "#6C63FF", "#ccc",  "rgba(108,99,255,0.3)"),
    "Light":  ("#ffffff", "#4C46CC", "#333",  "rgba(108,99,255,0.25)"),
    "Street": ("#f5f5f5", "#4C46CC", "#222",  "rgba(108,99,255,0.2)"),
}


# ══════════════════════════════════════════════════════════════════════════════
#  TIMELAPSE — CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

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

_TL_HEAT_SCALE = [
    [0.00, "#1E3A5F"],
    [0.30, "#1565C0"],
    [0.50, "#F57F17"],
    [0.75, "#E53935"],
    [1.00, "#7B1FA2"],
]


# ══════════════════════════════════════════════════════════════════════════════
#  HEATMAP — HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_hour(h: int) -> str:
    if h == 0:  return "12 AM"
    if h == 12: return "12 PM"
    return f"{h} AM" if h < 12 else f"{h - 12} PM"


def _safe(s, max_len: int = 60) -> str:
    return (
        str(s)[:max_len]
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


@st.cache_data(show_spinner=False)
def _station_opts(_df):
    return sorted(_df["police_station"].dropna().unique().tolist())


@st.cache_data(show_spinner=False)
def _vehicle_opts(_df):
    return sorted(_df["vehicle_type"].dropna().unique().tolist())


@st.cache_data(show_spinner=False)
def _compute_epi(_df_safe):
    if _df_safe.empty:
        return pd.DataFrame(
            columns=["police_station", "epi", "violation_count", "avg_cis", "lat", "lon"]
        )

    agg = (
        _df_safe.groupby("police_station", observed=True)
        .agg(
            violation_count=("cis", "count"),
            total_cis=("cis", "sum"),
            avg_cis=("cis", "mean"),
            lat=("latitude", "median"),
            lon=("longitude", "median"),
        )
        .reset_index()
    )

    def _norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)

    agg["epi"] = (
        0.4 * _norm(agg["total_cis"])
        + 0.3 * _norm(agg["violation_count"])
        + 0.3 * _norm(agg["avg_cis"])
    ) * 100

    return agg.sort_values("epi", ascending=False).head(_TOP_ENF_N).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _top_hotspots(_df_safe):
    if _df_safe.empty:
        return pd.DataFrame(columns=["location", "count", "avg_cis"])
    return (
        _df_safe.groupby("location", observed=True)
        .agg(count=("cis", "size"), avg_cis=("cis", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
        .head(_TOP_HOTSPOT_N)
    )


@st.cache_data(show_spinner=False)
def _violation_breakdown(_df_vtype):
    from data.helpers import parse_violations
    parsed = _df_vtype["violation_type"].dropna().astype(str).apply(parse_violations).explode()
    vc = parsed.dropna().value_counts().head(_TOP_VIOL_N).reset_index()
    vc.columns = ["violation_type", "count"]
    return vc


def _apply_filters(df, sel_stations, sel_vehicles, hour_range, sev_mode, day_type):
    out = df
    if sel_stations:
        out = out[out["police_station"].isin(sel_stations)]
    if sel_vehicles:
        out = out[out["vehicle_type"].isin(sel_vehicles)]
    h0, h1 = hour_range
    out = out[(out["hour"] >= h0) & (out["hour"] <= h1)]
    if sev_mode == "High Impact Only (CIS > 50)":
        out = out[out["cis"] > _HIGH_IMPACT_CIS]
    if day_type == "Weekdays Only":
        out = out[out["is_weekend"] == 0]
    elif day_type == "Weekends Only":
        out = out[out["is_weekend"] == 1]
    return out


def _layer_heatmap(m, heat_df):
    from folium.plugins import HeatMap

    valid = heat_df.dropna(subset=["latitude", "longitude", "cis"])
    if len(valid) > _MAX_HEAT_PTS:
        valid = valid.sample(_MAX_HEAT_PTS, random_state=42)
    if valid.empty:
        return

    cis_max = valid["cis"].max() or 1.0
    data = [
        [r["latitude"], r["longitude"], r["cis"] / cis_max]
        for _, r in valid[["latitude", "longitude", "cis"]].iterrows()
    ]
    HeatMap(data, radius=14, blur=12, max_zoom=15,
            gradient=_HEAT_GRADIENT, min_opacity=0.3).add_to(m)


def _layer_scatter(m, scatter_df, map_style: str = "Dark"):
    import folium

    bg, hd, tx, bd = _POPUP_STYLE[map_style]
    valid = scatter_df.dropna(subset=["latitude", "longitude", "cis"])
    top = valid.nlargest(_MAX_SCATTER_PTS, "cis")

    for _, r in top.iterrows():
        cis = r["cis"]
        color = "#F87171" if cis > 66 else "#FBBF24" if cis > 33 else "#34D399"

        loc   = _safe(r.get("location", "Unknown"), 60)
        vtype = _safe(r.get("vehicle_type", "N/A"), 30)
        stn   = _safe(r.get("police_station", "N/A"), 40)
        ts    = str(r.get("created_datetime", ""))[:16]

        viol_raw = r.get("violation_list", None)
        if isinstance(viol_raw, list):
            viol_str = _safe(", ".join(viol_raw[:2]), 70)
        else:
            viol_str = _safe(str(r.get("violation_type", "N/A")), 70)

        popup_html = (
            f"<div style='font-family:Inter,sans-serif;min-width:220px;"
            f"background:{bg};border-radius:8px;padding:12px;'>"
            f"<div style='color:{hd};font-size:12px;font-weight:700;"
            f"border-bottom:1px solid {bd};padding-bottom:6px;margin-bottom:8px;'>"
            f"Violation Detail</div>"
            f"<div style='color:{tx};font-size:11px;line-height:1.85;'>"
            f"<b>Location:</b> {loc}<br>"
            f"<b>Vehicle:</b> {vtype}<br>"
            f"<b>CIS Score:</b> <span style='color:{color};font-weight:700;'>{cis:.1f}/100</span><br>"
            f"<b>Offence:</b> {viol_str}<br>"
            f"<b>Station:</b> {stn}<br>"
            f"<b>Time:</b> {ts}"
            f"</div></div>"
        )

        folium.CircleMarker(
            location=[r["latitude"], r["longitude"]],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=1,
            popup=folium.Popup(popup_html, max_width=260),
        ).add_to(m)


def _layer_enforcement(m, epi_df, map_style: str = "Dark"):
    import folium

    bg, hd, tx, bd = _POPUP_STYLE[map_style]

    for rank, (_, row) in enumerate(epi_df.iterrows(), 1):
        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            continue

        stn = _safe(str(row["police_station"]), 50)
        popup_html = (
            f"<div style='font-family:Inter,sans-serif;min-width:210px;"
            f"background:{bg};border-radius:8px;padding:12px;'>"
            f"<div style='color:{hd};font-size:13px;font-weight:800;margin-bottom:8px;'>"
            f"#{rank} Enforcement Priority</div>"
            f"<div style='color:{tx};font-size:11px;line-height:1.9;'>"
            f"<b>Station:</b> {stn}<br>"
            f"<b>EPI Score:</b> <span style='color:{hd};font-weight:700;'>{row['epi']:.1f}</span>/100<br>"
            f"<b>Violations:</b> {int(row['violation_count']):,}<br>"
            f"<b>Avg CIS:</b> {row['avg_cis']:.1f}"
            f"</div></div>"
        )

        icon_html = (
            f"<div style='background:linear-gradient(135deg,#6C63FF,#00D2FF);"
            f"color:white;border-radius:50%;width:32px;height:32px;"
            f"display:flex;align-items:center;justify-content:center;"
            f"font-weight:800;font-size:13px;"
            f"box-shadow:0 0 12px rgba(108,99,255,0.6);"
            f"border:2px solid rgba(255,255,255,0.3);'>{rank}</div>"
        )

        folium.Marker(
            location=[row["lat"], row["lon"]],
            icon=folium.DivIcon(html=icon_html, icon_size=(32, 32), icon_anchor=(16, 16)),
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"#{rank} {row['police_station']} — EPI {row['epi']:.1f}",
        ).add_to(m)


def _layer_landmarks(m, map_style: str = "Dark"):
    import folium

    lm_bg    = "rgba(108,99,255,0.85)"  if map_style == "Dark"  else "rgba(76,70,204,0.9)"
    lm_color = "white"

    for lm in _LANDMARKS:
        html = (
            f'<div style="font-size:11px;color:{lm_color};'
            f'background:{lm_bg};'
            f'padding:2px 8px;border-radius:4px;'
            f'white-space:nowrap;font-weight:600;'
            f'box-shadow:0 2px 6px rgba(0,0,0,0.35);">'
            f'{lm["icon"]} {lm["name"]}</div>'
        )
        folium.Marker(
            location=[lm["lat"], lm["lon"]],
            popup=lm["name"],
            tooltip=lm["name"],
            icon=folium.DivIcon(html=html, icon_size=(160, 28), icon_anchor=(80, 14)),
        ).add_to(m)


_ANOMALY_PULSE_HTML = (
    "<style>"
    "@keyframes anomaly-pulse{"
    "0%{transform:scale(1);opacity:1;}"
    "50%{transform:scale(1.7);opacity:0.35;}"
    "100%{transform:scale(1);opacity:1;}"
    "}"
    ".anomaly-dot{"
    "animation:anomaly-pulse 1.5s ease-in-out infinite;"
    "width:16px;height:16px;border-radius:50%;"
    "background:#EF4444;border:2px solid #fff;"
    "box-shadow:0 0 10px 3px rgba(239,68,68,0.7);"
    "}"
    "</style>"
    "<div class='anomaly-dot'></div>"
)


def _layer_anomaly_markers(m) -> None:
    import folium
    flagged = st.session_state.get("anomaly_stations", [])
    if not flagged:
        return
    for stn in flagged:
        popup_html = (
            "<div style='font-family:Inter,sans-serif;min-width:180px;"
            "background:#1a1a2e;border-radius:8px;padding:10px;'>"
            "<div style='color:#EF4444;font-weight:700;font-size:12px;margin-bottom:6px;'>"
            "⚠️ Anomaly Detected</div>"
            "<div style='color:#ccc;font-size:11px;line-height:1.8;'>"
            f"<b>{stn['name']}</b><br>"
            "Anomaly detected — congestion spike."
            "</div></div>"
        )
        folium.CircleMarker(
            location=[stn["lat"], stn["lon"]],
            radius=18,
            color="#EF4444",
            fill=True,
            fill_color="#EF4444",
            fill_opacity=0.18,
            weight=2,
        ).add_to(m)
        folium.Marker(
            location=[stn["lat"], stn["lon"]],
            icon=folium.DivIcon(
                html=_ANOMALY_PULSE_HTML,
                icon_size=(16, 16),
                icon_anchor=(8, 8),
            ),
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"⚠️ {stn['name']} — Anomaly detected",
        ).add_to(m)


def _build_map(filtered_df, epi_df, map_layer, map_style: str = "Dark"):
    import folium

    m = folium.Map(
        location=_BGLR_CENTER,
        zoom_start=12,
        tiles=_MAP_TILES[map_style],
        control_scale=True,
    )

    if map_layer in ("Heatmap", "Both"):
        _layer_heatmap(m, filtered_df)

    if map_layer in ("Scatter Points", "Both"):
        _layer_scatter(m, filtered_df, map_style)

    _layer_enforcement(m, epi_df, map_style)
    _layer_landmarks(m, map_style)
    _layer_anomaly_markers(m)

    return m


def _render_stats_bar(filtered_df):
    total     = len(filtered_df)
    avg_cis   = filtered_df["cis"].mean() if total else 0.0
    rush_pct  = (filtered_df["is_rush_hour"].sum() / total * 100) if total else 0.0
    n_stn     = filtered_df["police_station"].nunique() if total else 0

    pills = [
        ("\U0001f4ca", "Filtered Records",  f"{total:,}"),
        ("\U0001f525", "Avg CIS Score",     f"{avg_cis:.1f}"),
        ("⚡",     "Rush Hour %",        f"{rush_pct:.1f}%"),
        ("\U0001f4cd", "Stations Affected",  str(n_stn)),
    ]

    for col, (icon, label, val) in zip(st.columns(4), pills):
        col.markdown(
            f"<div class='stat-pill'>"
            f"<div class='sp-icon'>{icon}</div>"
            f"<div class='sp-label'>{label}</div>"
            f"<div class='sp-val'>{val}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_map_legend():
    rank_badge = (
        "<span style='background:linear-gradient(135deg,#6C63FF,#00D2FF);"
        "color:white;border-radius:50%;padding:1px 6px;"
        "font-size:0.72rem;font-weight:700;'>1</span>"
    )
    lm_badge = (
        "<span style='background:rgba(0,0,0,0.65);"
        "border:1px solid rgba(255,255,255,0.15);"
        "border-radius:10px;padding:1px 7px;font-size:0.72rem;'>\U0001f534 Landmark</span>"
    )
    st.markdown(
        f"<div class='map-legend'>"
        f"<span style='color:#888;font-size:0.73rem;font-weight:700;"
        f"letter-spacing:.07em;text-transform:uppercase;'>Map Legend &nbsp;&nbsp;</span>"
        f"<span style='font-size:0.77rem;'>"
        f"<span style='color:#38BDF8;'>&#9632;</span> Low &nbsp;"
        f"<span style='color:#34D399;'>&#9632;</span> Moderate &nbsp;"
        f"<span style='color:#FBBF24;'>&#9632;</span> High &nbsp;"
        f"<span style='color:#F87171;'>&#9632;</span> Very High &nbsp;"
        f"<span style='color:#A78BFA;'>&#9632;</span> Critical"
        f"&nbsp;&nbsp;&nbsp;{rank_badge} Enforcement Zone"
        f"&nbsp;&nbsp;{lm_badge}"
        f"</span></div>",
        unsafe_allow_html=True,
    )


def _render_hotspots(filtered_df):
    st.markdown(
        "<div class='hmap-section-hd'>Top 10 Hotspot Locations</div>",
        unsafe_allow_html=True,
    )
    if filtered_df.empty:
        st.markdown(
            "<p style='color:#666;font-size:0.85rem;'>No data for selected filters.</p>",
            unsafe_allow_html=True,
        )
        return

    hs = _top_hotspots(filtered_df[["location", "cis"]].copy())
    rows_html = ""
    for i, (_, row) in enumerate(hs.iterrows(), 1):
        addr = str(row["location"])
        addr_disp = addr[:55] + "…" if len(addr) > 55 else addr
        rows_html += (
            f"<div class='hotspot-item'>"
            f"<div class='hs-rank'>{i}</div>"
            f"<div class='hs-addr'>{addr_disp}</div>"
            f"<div class='hs-stats'>"
            f"<div class='hs-count'>{int(row['count']):,} violations</div>"
            f"<div class='hs-cis'>CIS: {row['avg_cis']:.1f}</div>"
            f"</div></div>"
        )
    st.markdown(rows_html, unsafe_allow_html=True)


def _render_viol_chart(filtered_df):
    st.markdown(
        "<div class='hmap-section-hd'>Violation Breakdown for Filtered Data</div>",
        unsafe_allow_html=True,
    )
    if filtered_df.empty:
        st.markdown(
            "<p style='color:#666;font-size:0.85rem;'>No data for selected filters.</p>",
            unsafe_allow_html=True,
        )
        return

    vc = _violation_breakdown(filtered_df[["violation_type"]].copy())
    if vc.empty:
        return

    _label_map = {
        "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": "NEAR BUSTOP / SCHOOL",
        "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": "OPPOSITE PARKED VEHICLE",
        "PARKING IN A MAIN ROAD": "MAIN ROAD PARKING",
        "PARKING NEAR ROAD CROSSING": "NEAR ROAD CROSSING",
    }
    vc["label"] = vc["violation_type"].map(lambda v: _label_map.get(v, v))

    palette = ["#6C63FF", "#00D2FF", "#7C3AED", "#3B82F6", "#10B981", "#F59E0B"]

    fig = go.Figure(
        go.Bar(
            x=vc["count"].values[::-1],
            y=vc["label"].values[::-1],
            orientation="h",
            marker=dict(color=palette[: len(vc)], line=dict(width=0)),
            text=[f"{v:,}" for v in vc["count"].values[::-1]],
            textposition="outside",
            textfont=dict(size=10, color="#ccc"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=250,
        margin=dict(l=0, r=65, t=6, b=6),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=10, color="#aaa")),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TIMELAPSE — HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def format_hour(h: int) -> str:
    if h == 0:  return "12:00 AM (Midnight)"
    if h < 12:  return f"{h}:00 AM"
    if h == 12: return "12:00 PM (Noon)"
    return f"{h - 12}:00 PM"


@st.cache_data(show_spinner="⏳ Precomputing time-lapse data — runs once…")
def precompute_hourly_data(_df: pd.DataFrame) -> dict:
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


def _tl_render_hero():
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


def _tl_render_controls():
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


def _tl_render_hour_strip(selected_hour: int):
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


def _tl_render_live_stats(current: dict, daily_avg: float, city_avg_cis: float):
    hourly_avg = daily_avg / 24
    cis        = current["avg_cis"]
    cis_delta  = cis - city_avg_cis

    if cis > 60:   intensity = "🔴 CRITICAL"
    elif cis > 40: intensity = "🟠 HIGH"
    elif cis > 20: intensity = "🟡 MODERATE"
    else:          intensity = "🟢 LOW"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🚨 Violations This Hour", f"{current['count']:,}",
              delta=f"{current['count'] - hourly_avg:+.0f} vs hourly avg")
    c2.metric("🔥 Avg CIS Score", f"{cis:.1f}",
              delta=f"{cis_delta:+.1f} vs city avg")
    c3.metric("📍 Top Hotspot", current["top_station"],
              delta=f"{current['top_station_count']} violations here", delta_color="off")
    c4.metric("⚡ Intensity", intensity,
              delta=f"CIS {cis:.1f} / 100", delta_color="off")


def _tl_render_map(current: dict, selected_hour: int):
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
        color_continuous_scale=_TL_HEAT_SCALE,
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


def _tl_render_animation(hourly_data: dict):
    st.markdown("### 🎬 24-Hour Auto-Animation")
    st.markdown(
        "<p style='color:#888; font-size:0.88rem; margin-top:-8px; margin-bottom:12px;'>"
        "Click ▶ to watch violations build up and fade away — "
        "runs entirely in your browser with no page reloads</p>",
        unsafe_allow_html=True,
    )

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
        color_continuous_scale=_TL_HEAT_SCALE,
        range_color=[0, 100],
        opacity=0.85,
        title="Violation Density — 24-Hour Time-Lapse",
    )

    try:
        play_args = fig.layout.updatemenus[0].buttons[0].args[1]
        play_args["frame"]["duration"]      = 900
        play_args["transition"]["duration"] = 200
        for i, step in enumerate(fig.layout.sliders[0]["steps"]):
            step["label"] = format_hour(i)
    except (IndexError, KeyError, TypeError):
        pass

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


def _tl_render_comparison_chart(hourly_data: dict, selected_hour: int):
    st.markdown("### 📊 Violation Intensity — Full 24 Hours")

    tl_stats = pd.DataFrame({
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
        x=tl_stats["hour"],
        y=tl_stats["count"],
        marker_color=bar_colors,
        name="Violations",
        hovertemplate="Hour %{x}:00<br>Count: %{y:,}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=tl_stats["hour"],
        y=tl_stats["avg_cis"],
        mode="lines+markers",
        line=dict(color="#F59E0B", width=2),
        marker=dict(size=4),
        name="Avg CIS",
        yaxis="y2",
        hovertemplate="Hour %{x}:00<br>Avg CIS: %{y:.1f}<extra></extra>",
    ))

    fig.add_vline(x=selected_hour, line_dash="dash", line_color="#6C63FF", line_width=2)

    sel_count = int(tl_stats.loc[tl_stats["hour"] == selected_hour, "count"].iloc[0])
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
        yaxis=dict(title="Violations", gridcolor="rgba(255,255,255,0.05)"),
        yaxis2=dict(title="Avg CIS", overlaying="y", side="right", showgrid=False),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"tl_bar_{selected_hour}")


def _tl_render_insight(selected_hour: int):
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


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION RENDERS
# ══════════════════════════════════════════════════════════════════════════════

def _render_overview(df):
    st.markdown(
        "<p class='section-header'>Dashboard Overview</p>"
        "<p class='section-sub'>Key metrics from ~298K real BTP parking violation records</p>",
        unsafe_allow_html=True,
    )

    total_violations = len(df)
    avg_cis          = df["cis"].mean()
    worst_station    = df.groupby("police_station", observed=True)["cis"].mean().idxmax()
    worst_cis        = df.groupby("police_station", observed=True)["cis"].mean().max()
    peak_h           = int(df["hour"].value_counts().idxmax())

    if peak_h == 0:
        peak_str = "12:00 AM"
    elif peak_h < 12:
        peak_str = f"{peak_h}:00 AM"
    elif peak_h == 12:
        peak_str = "12:00 PM"
    else:
        peak_str = f"{peak_h - 12}:00 PM"

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""<div class='kpi-card'>
                <div class='kpi-label'>Total Violations</div>
                <div class='kpi-value'>{total_violations:,}</div>
                <div class='kpi-sub'>Jan – May records</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""<div class='kpi-card'>
                <div class='kpi-label'>Average CIS</div>
                <div class='kpi-value'>{avg_cis:.2f}</div>
                <div class='kpi-sub'>Congestion Impact Score</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""<div class='kpi-card'>
                <div class='kpi-label'>Worst Station</div>
                <div class='kpi-value' style='font-size:1.25rem;'>{worst_station}</div>
                <div class='kpi-sub'>Avg CIS {worst_cis:.1f}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""<div class='kpi-card'>
                <div class='kpi-label'>Peak Hour</div>
                <div class='kpi-value'>{peak_str}</div>
                <div class='kpi-sub'>Most violations recorded</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        sc = station_counts(df)
        n_bars = len(sc)
        bar_colors = [
            f"rgba({108 + int(i * 147 / n_bars)}, {99 - int(i * 20 / n_bars)}, 255, 0.85)"
            for i in range(n_bars)
        ][::-1]
        fig_stations = go.Figure(
            go.Bar(
                x=sc.values[::-1],
                y=sc.index[::-1],
                orientation="h",
                marker=dict(color=bar_colors, line=dict(width=0)),
                text=sc.values[::-1],
                textposition="outside",
                textfont=dict(size=11, color="#aaa"),
            )
        )
        fig_stations.update_layout(
            title=dict(text="Top 15 Police Stations by Violations", font=dict(size=15, color="#ddd")),
            xaxis_title=None,
            yaxis_title=None,
        )
        style_fig(fig_stations, 500)
        st.plotly_chart(fig_stations, use_container_width=True)

    with col_right:
        vc = vehicle_counts(df)
        fig_veh = go.Figure(
            go.Pie(
                labels=vc.index,
                values=vc.values,
                hole=0.52,
                marker=dict(colors=ACCENT_SEQ[: len(vc)]),
                textinfo="label+percent",
                textfont=dict(size=11, color="#ddd"),
                hoverinfo="label+value+percent",
            )
        )
        fig_veh.update_layout(
            title=dict(text="Vehicle Type Distribution", font=dict(size=15, color="#ddd")),
            showlegend=False,
        )
        style_fig(fig_veh, 500)
        st.plotly_chart(fig_veh, use_container_width=True)


def _render_heatmap(df):
    st.markdown(
        "<p class='section-header'>Congestion Heatmap — Bengaluru</p>"
        "<p class='section-sub'>CIS-weighted spatial density of parking violations "
        "— click markers for detail, use filters to drill into any slice</p>",
        unsafe_allow_html=True,
    )

    all_stations = _station_opts(df)
    all_vehicles = _vehicle_opts(df)

    with st.expander("\U0001f39b️ **Filters & Controls**", expanded=True):
        with st.form("hmap_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                sel_stations = st.multiselect(
                    "\U0001f4cd Police Station", all_stations, default=[],
                    placeholder="All stations", key="hmap_f_stations",
                )
                sev_mode = st.radio(
                    "Severity Filter",
                    ["All Violations", "High Impact Only (CIS > 50)"],
                    key="hmap_f_sev",
                )

            with col2:
                hour_range = st.slider(
                    "⏰ Time Range", min_value=0, max_value=23, value=(0, 23),
                    key="hmap_f_hours",
                )
                st.caption(f"From {_fmt_hour(hour_range[0])} → {_fmt_hour(hour_range[1])}")
                day_type = st.radio(
                    "Day Type", ["All Days", "Weekdays Only", "Weekends Only"],
                    key="hmap_f_day",
                )

            with col3:
                sel_vehicles = st.multiselect(
                    "\U0001f697 Vehicle Type", all_vehicles, default=[],
                    placeholder="All vehicle types", key="hmap_f_vehicles",
                )
                map_layer = st.radio(
                    "Map Layer", ["Heatmap", "Scatter Points", "Both"],
                    key="hmap_f_layer",
                )
                map_style = st.radio(
                    "Map Style", ["Dark", "Light", "Street"],
                    key="hmap_f_style",
                    help="Dark = CartoDB dark matter · Light = CartoDB positron · Street = OpenStreetMap",
                )

            st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
            st.form_submit_button("\U0001f504 Apply Filters", use_container_width=True, type="primary")

    filtered_df = _apply_filters(df, sel_stations, sel_vehicles, hour_range, sev_mode, day_type)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    _render_stats_bar(filtered_df)
    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='pw-info-banner'>"
        "ℹ️ <b>How to use:</b> Choose your Map Style (Dark / Light / Street) and Map Layer "
        "(Heatmap / Scatter / Both) in the filters above, then click "
        "<b>Apply Filters</b> to refresh the map. "
        "Click any marker or heat-point for violation details."
        "</div>",
        unsafe_allow_html=True,
    )

    try:
        from streamlit_folium import st_folium
    except ImportError:
        st.markdown(
            "<div class='pw-warn-banner'>⚠️ Map libraries not installed. "
            "Run: <code>pip install folium streamlit-folium</code></div>",
            unsafe_allow_html=True,
        )
        return

    if filtered_df.empty:
        st.markdown(
            "<div class='pw-empty-state'>"
            "<div class='es-icon'>🗺️</div>"
            "<div class='es-title'>No violations match your filters</div>"
            "<div class='es-sub'>"
            "Try widening the time range, clearing station/vehicle filters, "
            "or switching from <b>High Impact Only</b> to <b>All Violations</b>."
            "</div></div>",
            unsafe_allow_html=True,
        )
    else:
        epi_df = _compute_epi(
            filtered_df[["police_station", "cis", "latitude", "longitude"]].copy()
        )
        with st.spinner("\U0001f5fa️ Rendering map…"):
            m = _build_map(filtered_df, epi_df, map_layer, map_style)
            st_folium(m, width=None, height=580, returned_objects=[])

    _render_map_legend()

    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)
    left, right = st.columns([1, 1])
    with left:
        _render_hotspots(filtered_df)
    with right:
        _render_viol_chart(filtered_df)


def _render_timelapse(df: pd.DataFrame):
    with st.spinner("⏳ Precomputing hourly data…"):
        all_hourly   = precompute_hourly_data(df)
    daily_avg        = all_hourly["daily_avg"]
    city_avg_cis     = all_hourly["city_avg_cis"]

    if "hour_slider" not in st.session_state:
        st.session_state["hour_slider"] = 9

    _tl_render_hero()
    selected_hour, day_filter = _tl_render_controls()
    _tl_render_hour_strip(selected_hour)

    hourly_data = all_hourly[day_filter]
    current     = hourly_data[selected_hour]

    st.markdown("<hr style='border:1px solid rgba(255,255,255,0.07); margin:16px 0;'>",
                unsafe_allow_html=True)
    _tl_render_live_stats(current, daily_avg, city_avg_cis)

    st.markdown("<hr style='border:1px solid rgba(255,255,255,0.07); margin:16px 0;'>",
                unsafe_allow_html=True)

    map_tab, anim_tab = st.tabs(["🗺️ Selected Hour", "🎬 24-Hour Animation"])
    with map_tab:
        _tl_render_map(current, selected_hour)
    with anim_tab:
        _tl_render_animation(hourly_data)

    st.markdown("<hr style='border:1px solid rgba(255,255,255,0.07); margin:16px 0;'>",
                unsafe_allow_html=True)
    _tl_render_comparison_chart(hourly_data, selected_hour)
    _tl_render_insight(selected_hour)


# ══════════════════════════════════════════════════════════════════════════════
#  DIRECT MAP (no filter UI)
# ══════════════════════════════════════════════════════════════════════════════

def _render_map_direct(df: pd.DataFrame, map_mode: str) -> None:
    try:
        import folium
        from folium.plugins import HeatMap
        from streamlit_folium import st_folium
    except ImportError:
        st.warning("Run: `pip install folium streamlit-folium`")
        return

    valid = df.dropna(subset=["latitude", "longitude", "cis"])
    if len(valid) > _MAX_HEAT_PTS:
        valid = valid.sample(_MAX_HEAT_PTS, random_state=42)

    if map_mode == "CIS Score":
        cis_max = float(valid["cis"].max()) or 1.0
        weights = (valid["cis"] / cis_max).tolist()
    else:
        weights = [1.0] * len(valid)

    data = list(zip(valid["latitude"].tolist(), valid["longitude"].tolist(), weights))

    m = folium.Map(
        location=_BGLR_CENTER, zoom_start=12,
        tiles=_MAP_TILES["Dark"], control_scale=True,
    )
    HeatMap(data, radius=14, blur=12, max_zoom=15,
            gradient=_HEAT_GRADIENT, min_opacity=0.3).add_to(m)

    epi_df = _compute_epi(df[["police_station", "cis", "latitude", "longitude"]].copy())
    _layer_enforcement(m, epi_df)
    _layer_landmarks(m)
    _layer_anomaly_markers(m)

    st_folium(m, width=None, height=580, returned_objects=[])
    _render_map_legend()


# ══════════════════════════════════════════════════════════════════════════════
#  CRITICAL JUNCTION ALERTS (expander content)
# ══════════════════════════════════════════════════════════════════════════════

def _render_critical_alerts(df: pd.DataFrame) -> None:
    # Top stations + vehicle breakdown charts (from overview)
    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        sc = station_counts(df)
        n_bars = len(sc)
        bar_colors = [
            f"rgba({108 + int(i * 147 / n_bars)}, {99 - int(i * 20 / n_bars)}, 255, 0.85)"
            for i in range(n_bars)
        ][::-1]
        fig_stations = go.Figure(
            go.Bar(
                x=sc.values[::-1],
                y=sc.index[::-1],
                orientation="h",
                marker=dict(color=bar_colors, line=dict(width=0)),
                text=sc.values[::-1],
                textposition="outside",
                textfont=dict(size=11, color="#aaa"),
            )
        )
        fig_stations.update_layout(
            title=dict(text="Top 15 Police Stations by Violations", font=dict(size=15, color="#ddd")),
            xaxis_title=None,
            yaxis_title=None,
        )
        style_fig(fig_stations, 500)
        st.plotly_chart(fig_stations, use_container_width=True)

    with col_right:
        vc = vehicle_counts(df)
        fig_veh = go.Figure(
            go.Pie(
                labels=vc.index,
                values=vc.values,
                hole=0.52,
                marker=dict(colors=ACCENT_SEQ[: len(vc)]),
                textinfo="label+percent",
                textfont=dict(size=11, color="#ddd"),
                hoverinfo="label+value+percent",
            )
        )
        fig_veh.update_layout(
            title=dict(text="Vehicle Type Distribution", font=dict(size=15, color="#ddd")),
            showlegend=False,
        )
        style_fig(fig_veh, 500)
        st.plotly_chart(fig_veh, use_container_width=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        _render_hotspots(df)
    with right:
        _render_viol_chart(df)


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def render(df):
    # ── KPI cards ────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total violations", f"{len(df):,}")
    with col2:
        st.metric("Avg CIS", f"{df['cis'].mean():.2f}")
    with col3:
        st.metric("Worst zone", df.groupby("police_station", observed=True)["cis"].mean().idxmax())
    with col4:
        peak_hr = int(df.groupby("hour")["cis"].mean().idxmax())
        suffix = "AM" if peak_hr < 12 else "PM"
        display_hr = peak_hr if peak_hr <= 12 else peak_hr - 12
        st.metric("Peak hour", f"{display_hr}:00 {suffix}")

    # ── Count vs CIS toggle ───────────────────────────────────────────────────
    map_mode = st.radio(
        "Heatmap weight",
        ["CIS Score", "Violation Count"],
        horizontal=True,
        key="cc_map_mode",
        label_visibility="collapsed",
    )

    # ── Map (direct, no expander) ─────────────────────────────────────────────
    _render_map_direct(df, map_mode)

    st.divider()

    with st.expander("⏳ Hour-by-hour animation", expanded=False):
        _render_timelapse(df)

    with st.expander("🔴 Critical junction alerts", expanded=False):
        _render_critical_alerts(df)
