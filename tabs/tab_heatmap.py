"""
Premium Heatmap Tab — 4-layer folium map (heatmap, scatter, enforcement markers,
landmarks), filter panel with submit, stats bar, and below-map insights panel.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_BGLR_CENTER = [12.9716, 77.5946]
_MAX_HEAT_PTS = 25_000
_MAX_SCATTER_PTS = 2_000
_TOP_ENF_N = 10
_TOP_HOTSPOT_N = 10
_TOP_VIOL_N = 6
_HIGH_IMPACT_CIS = 50.0

_HEAT_GRADIENT = {
    0.2: "#3B82F6",
    0.4: "#10B981",
    0.6: "#F59E0B",
    0.8: "#EF4444",
    1.0: "#7C3AED",
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

# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════

_CSS = """<style>
.stat-pill {
    background: rgba(30,33,48,0.65);
    border: 1px solid rgba(108,99,255,0.15);
    border-radius: 14px;
    backdrop-filter: blur(12px);
    padding: 18px 14px 14px;
    text-align: center;
    margin-bottom: 6px;
}
.stat-pill .sp-icon { font-size: 1.25rem; }
.stat-pill .sp-label {
    color: #888; font-size: 0.71rem;
    letter-spacing: .07em; text-transform: uppercase; margin: 4px 0 2px;
}
.stat-pill .sp-val {
    font-size: 1.55rem; font-weight: 800;
    background: linear-gradient(90deg, #6C63FF, #00D2FF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hmap-section-hd {
    color: #aaa; font-size: 0.77rem; font-weight: 700;
    letter-spacing: .09em; text-transform: uppercase; margin: 22px 0 10px;
}
.hotspot-item {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 13px; margin-bottom: 7px;
    background: rgba(30,33,48,0.65);
    border: 1px solid rgba(108,99,255,0.15);
    border-radius: 10px;
    backdrop-filter: blur(12px);
}
.hs-rank {
    min-width: 26px; height: 26px; line-height: 26px; text-align: center;
    border-radius: 50%; font-size: 0.72rem; font-weight: 700;
    background: linear-gradient(135deg, #6C63FF, #00D2FF); color: #fff;
    flex-shrink: 0;
}
.hs-addr { flex: 1; color: #ccc; font-size: 0.8rem; line-height: 1.3; }
.hs-stats { text-align: right; white-space: nowrap; flex-shrink: 0; }
.hs-count { color: #6C63FF; font-weight: 700; font-size: 0.85rem; }
.hs-cis   { color: #888; font-size: 0.7rem; margin-top: 2px; }
.map-legend {
    margin-top: 14px; padding: 10px 16px;
    background: rgba(30,33,48,0.65);
    border: 1px solid rgba(108,99,255,0.12);
    border-radius: 10px; backdrop-filter: blur(12px);
}
</style>"""


# ══════════════════════════════════════════════════════════════════════════════
#  PURE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_hour(h: int) -> str:
    if h == 0:  return "12 AM"
    if h == 12: return "12 PM"
    return f"{h} AM" if h < 12 else f"{h - 12} PM"


def _safe(s, max_len: int = 60) -> str:
    """HTML-escape and truncate a value for use in popup markup."""
    return (
        str(s)[:max_len]
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CACHED AGGREGATIONS  (pass only hashable column subsets — no list columns)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _station_opts(_df):
    return sorted(_df["police_station"].dropna().unique().tolist())


@st.cache_data(show_spinner=False)
def _vehicle_opts(_df):
    return sorted(_df["vehicle_type"].dropna().unique().tolist())


@st.cache_data(show_spinner=False)
def _compute_epi(_df_safe):
    """
    EPI per station.
    _df_safe must contain: police_station, cis, latitude, longitude
    (no list-type columns so @st.cache_data can hash it cleanly).
    """
    if _df_safe.empty:
        return pd.DataFrame(
            columns=["police_station", "epi", "violation_count", "avg_cis", "lat", "lon"]
        )

    agg = (
        _df_safe.groupby("police_station")
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
    """Top locations by violation count + avg CIS. _df_safe: [location, cis]."""
    if _df_safe.empty:
        return pd.DataFrame(columns=["location", "count", "avg_cis"])
    return (
        _df_safe.groupby("location")
        .agg(count=("cis", "size"), avg_cis=("cis", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
        .head(_TOP_HOTSPOT_N)
    )


@st.cache_data(show_spinner=False)
def _violation_breakdown(_df_vtype):
    """Count top violation types. _df_vtype: DataFrame with [violation_type] string column."""
    from data.helpers import parse_violations
    parsed = _df_vtype["violation_type"].dropna().apply(parse_violations).explode()
    vc = parsed.dropna().value_counts().head(_TOP_VIOL_N).reset_index()
    vc.columns = ["violation_type", "count"]
    return vc


# ══════════════════════════════════════════════════════════════════════════════
#  FILTER LOGIC
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  MAP LAYER BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

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


def _layer_scatter(m, scatter_df):
    import folium

    valid = scatter_df.dropna(subset=["latitude", "longitude", "cis"])
    top = valid.nlargest(_MAX_SCATTER_PTS, "cis")

    for _, r in top.iterrows():
        cis = r["cis"]
        color = "#EF4444" if cis > 66 else "#F59E0B" if cis > 33 else "#10B981"

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
            "<div style='font-family:Inter,sans-serif;min-width:220px;"
            "background:#1a1a2e;border-radius:8px;padding:12px;'>"
            "<div style='color:#6C63FF;font-size:12px;font-weight:700;"
            "border-bottom:1px solid rgba(108,99,255,0.3);padding-bottom:6px;margin-bottom:8px;'>"
            "Violation Detail</div>"
            f"<div style='color:#ccc;font-size:11px;line-height:1.85;'>"
            f"<b>Location:</b> {loc}<br>"
            f"<b>Vehicle:</b> {vtype}<br>"
            f"<b>CIS Score:</b> <span style='color:{color};font-weight:700;'>{cis:.1f}/100</span><br>"
            f"<b>Offence:</b> {viol_str}<br>"
            f"<b>Station:</b> {stn}<br>"
            f"<b>Time:</b> {ts}"
            "</div></div>"
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


def _layer_enforcement(m, epi_df):
    import folium

    for rank, (_, row) in enumerate(epi_df.iterrows(), 1):
        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            continue

        stn = _safe(str(row["police_station"]), 50)
        popup_html = (
            "<div style='font-family:Inter,sans-serif;min-width:210px;"
            "background:#1a1a2e;border-radius:8px;padding:12px;'>"
            f"<div style='color:#6C63FF;font-size:13px;font-weight:800;margin-bottom:8px;'>"
            f"#{rank} Enforcement Priority</div>"
            f"<div style='color:#ccc;font-size:11px;line-height:1.9;'>"
            f"<b>Station:</b> {stn}<br>"
            f"<b>EPI Score:</b> <span style='color:#6C63FF;font-weight:700;'>{row['epi']:.1f}</span>/100<br>"
            f"<b>Violations:</b> {int(row['violation_count']):,}<br>"
            f"<b>Avg CIS:</b> {row['avg_cis']:.1f}"
            "</div></div>"
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


def _layer_landmarks(m):
    import folium

    for lm in _LANDMARKS:
        html = (
            f'<div style="font-size:11px;color:white;'
            f'background:rgba(108,99,255,0.8);'
            f'padding:2px 8px;border-radius:4px;'
            f'white-space:nowrap;font-weight:600;'
            f'box-shadow:0 2px 6px rgba(0,0,0,0.4);">'
            f'{lm["icon"]} {lm["name"]}</div>'
        )
        folium.Marker(
            location=[lm["lat"], lm["lon"]],
            popup=lm["name"],
            tooltip=lm["name"],
            icon=folium.DivIcon(html=html, icon_size=(160, 28), icon_anchor=(80, 14)),
        ).add_to(m)


def _build_map(filtered_df, epi_df, map_layer):
    import folium

    m = folium.Map(
        location=_BGLR_CENTER,
        zoom_start=12,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    if map_layer in ("Heatmap", "Both"):
        _layer_heatmap(m, filtered_df)

    if map_layer in ("Scatter Points", "Both"):
        _layer_scatter(m, filtered_df)

    _layer_enforcement(m, epi_df)  # always shown
    _layer_landmarks(m)             # always shown

    return m


# ══════════════════════════════════════════════════════════════════════════════
#  STATS BAR
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  MAP LEGEND
# ══════════════════════════════════════════════════════════════════════════════

def _render_legend():
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
        f"<span style='color:#3B82F6;'>&#9632;</span> Low &nbsp;"
        f"<span style='color:#10B981;'>&#9632;</span> Moderate &nbsp;"
        f"<span style='color:#F59E0B;'>&#9632;</span> High &nbsp;"
        f"<span style='color:#EF4444;'>&#9632;</span> Very High &nbsp;"
        f"<span style='color:#7C3AED;'>&#9632;</span> Critical"
        f"&nbsp;&nbsp;&nbsp;{rank_badge} Enforcement Zone"
        f"&nbsp;&nbsp;{lm_badge}"
        f"</span></div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  INSIGHTS PANEL
# ══════════════════════════════════════════════════════════════════════════════

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

    # Shorten long labels for readability
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
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def render(df):
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown(
        "<p class='section-header'>Congestion Heatmap — Bengaluru</p>"
        "<p class='section-sub'>CIS-weighted spatial density of parking violations "
        "— click markers for detail, use filters to drill into any slice</p>",
        unsafe_allow_html=True,
    )

    # ── Section 1: Filter Panel ────────────────────────────────────────────────
    all_stations = _station_opts(df)
    all_vehicles = _vehicle_opts(df)

    with st.expander("\U0001f39b️ **Filters & Controls**", expanded=True):
        with st.form("hmap_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                sel_stations = st.multiselect(
                    "\U0001f4cd Police Station",
                    all_stations,
                    default=[],
                    placeholder="All stations",
                    key="hmap_f_stations",
                )
                sev_mode = st.radio(
                    "Severity Filter",
                    ["All Violations", "High Impact Only (CIS > 50)"],
                    key="hmap_f_sev",
                )

            with col2:
                hour_range = st.slider(
                    "⏰ Time Range",
                    min_value=0, max_value=23, value=(0, 23),
                    key="hmap_f_hours",
                )
                st.caption(
                    f"From {_fmt_hour(hour_range[0])} → {_fmt_hour(hour_range[1])}"
                )
                day_type = st.radio(
                    "Day Type",
                    ["All Days", "Weekdays Only", "Weekends Only"],
                    key="hmap_f_day",
                )

            with col3:
                sel_vehicles = st.multiselect(
                    "\U0001f697 Vehicle Type",
                    all_vehicles,
                    default=[],
                    placeholder="All vehicle types",
                    key="hmap_f_vehicles",
                )
                map_layer = st.radio(
                    "Map Layer",
                    ["Heatmap", "Scatter Points", "Both"],
                    key="hmap_f_layer",
                )

            st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
            st.form_submit_button(
                "\U0001f504 Apply Filters",
                use_container_width=True,
                type="primary",
            )

    # ── Apply filters ──────────────────────────────────────────────────────────
    filtered_df = _apply_filters(
        df, sel_stations, sel_vehicles, hour_range, sev_mode, day_type
    )

    # ── Section 2: Stats Bar ───────────────────────────────────────────────────
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    _render_stats_bar(filtered_df)
    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    # ── Section 3: Map ─────────────────────────────────────────────────────────
    try:
        from streamlit_folium import st_folium
    except ImportError:
        st.warning(
            "Install folium and streamlit-folium to render the map:\n\n"
            "`pip install folium streamlit-folium`"
        )
        return

    if filtered_df.empty:
        st.markdown(
            "<div style='text-align:center;padding:80px 20px;"
            "background:rgba(30,33,48,0.65);border-radius:14px;"
            "border:1px solid rgba(108,99,255,0.15);'>"
            "<div style='font-size:2rem;'>\U0001f5fa️</div>"
            "<div style='color:#888;margin-top:10px;font-size:1rem;'>"
            "No records match the selected filters.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        # Only non-list columns passed to the cached EPI function
        epi_df = _compute_epi(
            filtered_df[["police_station", "cis", "latitude", "longitude"]].copy()
        )

        with st.spinner("\U0001f5fa️ Rendering map…"):
            m = _build_map(filtered_df, epi_df, map_layer)
            st_folium(m, width=None, height=580, returned_objects=[])

    _render_legend()

    # ── Section 4: Insights Panel ──────────────────────────────────────────────
    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)
    left, right = st.columns([1, 1])
    with left:
        _render_hotspots(filtered_df)
    with right:
        _render_viol_chart(filtered_df)
