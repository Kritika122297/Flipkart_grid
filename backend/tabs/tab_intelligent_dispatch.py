"""
tabs/tab_intelligent_dispatch.py — 🎯 Intelligent Dispatch
Merged from: tab_enforcement + tab_simulator + tab_compare
"""
from __future__ import annotations
import io
import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from charts.utils import enforcement_table, style_fig, PLOTLY_TEMPLATE, CHART_BG
from data.loader import load_and_process_data


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATOR — CONSTANTS & CSS
# ══════════════════════════════════════════════════════════════════════════════

_STRATEGIES = [
    "Top Stations by EPI (Recommended)",
    "High Rush Hour Zones",
    "Heavy Vehicle Hotspots",
    "Junction-Heavy Areas",
]

_STRATEGY_SORT = {
    "Top Stations by EPI (Recommended)": "epi",
    "High Rush Hour Zones":              "rush_pct",
    "Heavy Vehicle Hotspots":            "heavy_pct",
    "Junction-Heavy Areas":              "junction_pct",
}

_TIME_OPTIONS = [
    "All Day",
    "Rush Hour Only (7–10AM, 4–8PM)",
    "Morning Rush Only (7–10AM)",
    "Evening Rush Only (4–8PM)",
]

_TIME_MULTIPLIER = {
    "All Day":                              1.00,
    "Rush Hour Only (7–10AM, 4–8PM)": 0.55,
    "Morning Rush Only (7–10AM)":      0.30,
    "Evening Rush Only (4–8PM)":       0.25,
}

_SIM_CSS = """<style>
.sim-card {
    background: linear-gradient(135deg, rgba(0,210,100,0.15), rgba(0,210,255,0.08));
    backdrop-filter: blur(16px);
    border: 1px solid rgba(0,210,100,0.25);
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 4px;
}
.sim-card .sc-label { font-size: 0.78rem; font-weight: 600; color: #888;
    text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 6px; }
.sim-card .sc-val {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #00D264, #00D2FF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.15;
}
.sim-card .sc-sub { font-size: 0.75rem; color: #666; margin-top: 4px; }
.pro-tip-box {
    background: rgba(108,99,255,0.08);
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 14px;
    padding: 22px 20px;
    backdrop-filter: blur(12px);
    height: 100%;
    box-sizing: border-box;
}
.pro-tip-box .pt-title { font-size: 0.88rem; font-weight: 700; color: #00D2FF;
    margin-bottom: 14px; letter-spacing: 0.05em; }
.pro-tip-box .pt-body { font-size: 0.84rem; color: #ccc; line-height: 1.75; }
.pro-tip-box .pt-hi { color: #00D264; font-weight: 700; }
.sim-section-hd {
    color: #aaa; font-size: 0.77rem; font-weight: 700;
    letter-spacing: .09em; text-transform: uppercase;
    margin: 24px 0 12px;
}
div.stFormSubmitButton > button {
    background: linear-gradient(135deg, #6C63FF, #00D2FF) !important;
    color: white !important; font-weight: 700 !important;
    font-size: 1rem !important; border: none !important;
    border-radius: 10px !important; padding: 12px !important;
    transition: opacity 0.2s, box-shadow 0.2s;
}
div.stFormSubmitButton > button:hover {
    opacity: 0.9 !important;
    box-shadow: 0 4px 20px rgba(108,99,255,0.45) !important;
}
</style>"""


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARE — CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_A = "#6C63FF"
_B = "#00D264"


# ══════════════════════════════════════════════════════════════════════════════
#  ENFORCEMENT — HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _nearest_neighbor(coords: list[tuple]) -> list[int]:
    n = len(coords)
    if n == 0:
        return []
    visited = [False] * n
    order   = [0]
    visited[0] = True
    for _ in range(n - 1):
        last_lat, last_lon = coords[order[-1]]
        best_d, best_i = float("inf"), -1
        for i in range(n):
            if not visited[i]:
                dlat = coords[i][0] - last_lat
                dlon = coords[i][1] - last_lon
                d = dlat * dlat + dlon * dlon
                if d < best_d:
                    best_d, best_i = d, i
        if best_i >= 0:
            order.append(best_i)
            visited[best_i] = True
    return order


@st.cache_data(show_spinner=False)
def _build_patrol_route(_df: pd.DataFrame, top_n: int = 10):
    enf = enforcement_table(_df)
    coords_df = (
        _df.groupby("police_station", observed=True)
        .agg(lat=("latitude", "median"), lon=("longitude", "median"))
        .reset_index()
    )
    merged = pd.merge(enf.head(top_n), coords_df, on="police_station", how="inner")
    merged = merged.dropna(subset=["lat", "lon"])
    if merged.empty:
        return pd.DataFrame()

    pts   = list(zip(merged["lat"], merged["lon"]))
    order = _nearest_neighbor(pts)
    return merged.iloc[order].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATOR — HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Computing station profiles…")
def compute_epi(_df):
    agg = _df.groupby("police_station", observed=True).agg(
        total_cis=("cis", "sum"),
        avg_cis=("cis", "mean"),
        count=("cis", "size"),
        rush_pct=("is_rush_hour", "mean"),
        heavy_pct=("vehicle_size_score", lambda x: (x >= 7).mean()),
        junction_pct=("junction_factor", lambda x: (x == 2.0).mean()),
        avg_lat=("latitude", "mean"),
        avg_lon=("longitude", "mean"),
    ).reset_index()

    def _norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) * 100 if mx > mn else pd.Series(50.0, index=s.index)

    agg["epi"] = (
        0.4 * _norm(agg["total_cis"])
        + 0.3 * _norm(agg["count"])
        + 0.3 * _norm(agg["avg_cis"])
    )
    agg["epi"] = _norm(agg["epi"])
    return agg.sort_values("epi", ascending=False).reset_index(drop=True)


def run_simulation(
    station_df: pd.DataFrame,
    n_teams: int,
    effectiveness_pct: int,
    strategy: str,
    time_focus: str,
    value_of_time: int,
    vehicles_per_hour: int,
) -> pd.DataFrame:
    n = min(n_teams, len(station_df))
    sort_col = _STRATEGY_SORT.get(strategy, "epi")
    selected = station_df.sort_values(sort_col, ascending=False).head(n).copy()

    eff    = effectiveness_pct / 100.0
    t_mult = _TIME_MULTIPLIER.get(time_focus, 1.0)

    selected["violations_targeted"] = (selected["count"] * t_mult).astype(int)
    selected["violations_reduced"]  = (selected["violations_targeted"] * eff).astype(int)
    selected["cis_reduced"]         = (selected["avg_cis"] * eff).round(1)

    avg_delay_min = selected["avg_cis"] / 100.0 * 5.0
    selected["daily_savings_rs"] = (
        selected["violations_reduced"]
        * avg_delay_min
        * (vehicles_per_hour / 60.0)
        * value_of_time
    ).astype(int)
    selected["annual_savings_rs"] = selected["daily_savings_rs"] * 365

    return selected


def _sim_phase1_metrics(results: pd.DataFrame, station_df: pd.DataFrame) -> None:
    cards = [
        ("Violations Prevented",   f"{results['violations_reduced'].sum():,}",       "violations prevented daily"),
        ("CIS Reduction",          f"{results['cis_reduced'].mean():.1f}%",           "avg congestion impact drop"),
        ("Annual Savings",         f"₹{results['annual_savings_rs'].sum() / 1e7:.1f} Cr", "estimated annual savings"),
        ("Coverage",               f"{len(results)} / {len(station_df)}",             "stations covered"),
    ]
    for col, (label, value, sub) in zip(st.columns(4), cards):
        col.markdown(
            f"<div class='sim-card'>"
            f"<div class='sc-label'>{label}</div>"
            f"<div class='sc-val'>{value}</div>"
            f"<div class='sc-sub'>{sub}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _sim_phase2_before_after(results: pd.DataFrame) -> None:
    top    = results.head(10)
    labels = [(s[:28] + "…" if len(s) > 28 else s) for s in top["police_station"].tolist()]
    before = top["violations_targeted"].tolist()
    after  = (top["violations_targeted"] - top["violations_reduced"]).tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Before Enforcement", x=labels, y=before,
                         marker_color="rgba(239,68,68,0.8)", marker_line_width=0))
    fig.add_trace(go.Bar(name="After Enforcement",  x=labels, y=after,
                         marker_color="rgba(0,210,100,0.8)",  marker_line_width=0))
    fig.update_layout(
        title=dict(text="Violation Count: Before vs After Enforcement",
                   font=dict(size=15, color="#ddd")),
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=20, r=20, t=50, b=100),
        legend=dict(orientation="h", y=-0.32, x=0.5, xanchor="center",
                    font=dict(size=12, color="#ccc")),
        xaxis=dict(tickfont=dict(size=9, color="#aaa"), tickangle=-30),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _sim_phase3_table(results: pd.DataFrame) -> None:
    display = results[[
        "police_station", "violations_targeted", "violations_reduced",
        "cis_reduced", "daily_savings_rs", "annual_savings_rs",
    ]].copy()
    display.insert(0, "Rank", range(1, len(display) + 1))

    st.dataframe(
        display,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "police_station": st.column_config.TextColumn("\U0001f4cd Station", width="medium"),
            "violations_targeted": st.column_config.NumberColumn("Violations Targeted", format="%d"),
            "violations_reduced":  st.column_config.NumberColumn("Expected Reduction",  format="%d"),
            "cis_reduced":         st.column_config.ProgressColumn("CIS Drop", min_value=0, max_value=100, format="%.1f"),
            "daily_savings_rs":    st.column_config.NumberColumn("Daily Savings (₹)",   format="₹%d"),
            "annual_savings_rs":   st.column_config.NumberColumn("Annual Savings (₹)",  format="₹%d"),
        },
        use_container_width=True,
        hide_index=True,
        height=420,
    )


def _sim_phase4_strategy_comparison(
    results, station_df, n_teams, effectiveness_pct,
    time_focus, value_of_time, vehicles_per_hour, strategy,
) -> None:
    left, right = st.columns([1.15, 0.85])

    with left:
        st.markdown("<div class='sim-section-hd'>\U0001f4ca Annual Savings by Station</div>",
                    unsafe_allow_html=True)
        top8 = results.nlargest(8, "annual_savings_rs")
        labels = [(s[:24] + "…" if len(s) > 24 else s) for s in top8["police_station"]]
        palette = [
            f"rgba({max(108 - i*7, 40)},{min(99 + i*10, 220)},{max(255 - i*12, 80)},0.85)"
            for i in range(len(top8))
        ]
        fig2 = go.Figure(go.Bar(
            x=top8["annual_savings_rs"].values / 1e5,
            y=labels,
            orientation="h",
            marker=dict(color=palette, line=dict(width=0)),
            text=[f"₹{v/1e5:.1f}L" for v in top8["annual_savings_rs"].values],
            textposition="outside",
            textfont=dict(size=10, color="#ccc"),
        ))
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=0, r=75, t=6, b=10),
            xaxis=dict(showgrid=False, showticklabels=False, title="₹ Lakhs / year"),
            yaxis=dict(showgrid=False, tickfont=dict(size=9, color="#aaa")),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.markdown("<div class='sim-section-hd'>\U0001f4a1 Strategy Intelligence</div>",
                    unsafe_allow_html=True)
        total_viol   = station_df["count"].sum()
        sel_coverage = results["count"].sum()
        sel_pct      = sel_coverage / total_viol * 100 if total_viol else 0.0
        cur_savings  = results["annual_savings_rs"].sum()

        epi_sim      = run_simulation(station_df, n_teams, effectiveness_pct,
                                      "Top Stations by EPI (Recommended)",
                                      time_focus, value_of_time, vehicles_per_hour)
        epi_savings  = epi_sim["annual_savings_rs"].sum()
        epi_coverage = epi_sim["count"].sum() / total_viol * 100 if total_viol else 0.0
        diff_cr      = (epi_savings - cur_savings) / 1e7

        if strategy == "Top Stations by EPI (Recommended)":
            body = (
                f"Your strategy covers <span class='pt-hi'>{sel_pct:.1f}%</span> "
                f"of total violations.<br><br>"
                f"You're already using the <span class='pt-hi'>optimal EPI strategy</span>! "
                f"This deployment targets the stations with the highest combined CIS "
                f"volume, frequency, and severity."
            )
        elif diff_cr > 0.01:
            body = (
                f"Your strategy covers <span class='pt-hi'>{sel_pct:.1f}%</span> "
                f"of total violations.<br><br>"
                f"EPI top zones would cover <span class='pt-hi'>{epi_coverage:.1f}%</span> instead.<br><br>"
                f"Switching to EPI strategy saves an additional<br>"
                f"<span class='pt-hi'>₹{diff_cr:.1f} Cr</span> annually "
                f"with the same {n_teams} teams."
            )
        else:
            body = (
                f"Your strategy covers <span class='pt-hi'>{sel_pct:.1f}%</span> "
                f"of total violations.<br><br>"
                f"This performs comparably to the EPI strategy — "
                f"a good choice given the current data distribution."
            )

        st.markdown(
            f"<div class='pro-tip-box'>"
            f"<div class='pt-title'>\U0001f4a1 Strategy Intelligence</div>"
            f"<div class='pt-body'>{body}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _sim_generate_report(results, strategy, n_teams, effectiveness_pct, time_focus, value_of_time) -> str:
    buf = io.StringIO()
    sep = "-" * 52
    buf.write("PARKWATCH AI — ENFORCEMENT SIMULATION REPORT\n")
    buf.write("=" * 52 + "\n")
    buf.write(f"Generated    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    buf.write(f"Strategy     : {strategy}\n")
    buf.write(f"Teams        : {n_teams}\n")
    buf.write(f"Effectiveness: {effectiveness_pct}%\n")
    buf.write(f"Time Focus   : {time_focus}\n")
    buf.write(f"Value of Time: ₹{value_of_time}/hr\n")
    buf.write(sep + "\n")
    buf.write("PROJECTED RESULTS\n")
    buf.write(sep + "\n")
    buf.write(f"Violations reduced daily  : {results['violations_reduced'].sum():,}\n")
    buf.write(f"Avg CIS reduction         : {results['cis_reduced'].mean():.1f}%\n")
    buf.write(f"Annual savings            : ₹{results['annual_savings_rs'].sum()/1e7:.2f} Cr\n")
    buf.write(f"Stations covered          : {len(results)}\n")
    buf.write(sep + "\n")
    buf.write("DEPLOYMENT PLAN\n")
    buf.write(sep + "\n")
    buf.write(f"{'Rank':<5} {'Station':<42} {'Reduce':>7}  {'Annual Savings':>15}\n")
    buf.write(sep + "\n")
    for rank, (_, row) in enumerate(results.iterrows(), 1):
        buf.write(
            f"{rank:<5} {row['police_station'][:42]:<42} "
            f"{int(row['violations_reduced']):>7,}  "
            f"₹{int(row['annual_savings_rs']):>13,}\n"
        )
    buf.write("=" * 52 + "\n")
    buf.write("ParkWatch AI — Flipkart Gridlock Hackathon 2.0\n")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARE — HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _load_uploaded(raw: bytes, key: str):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(raw)
        path = tmp.name
    df, stats = load_and_process_data(path)
    try:
        os.unlink(path)
    except Exception:
        pass
    return df, stats


@st.cache_data(show_spinner=False)
def _cmp_top_stations(_df: pd.DataFrame, n: int = 15):
    return (
        _df.groupby("police_station", observed=True)
        .agg(violations=("cis", "size"), avg_cis=("cis", "mean"))
        .reset_index()
        .sort_values("violations", ascending=False)
        .head(n)
    )


@st.cache_data(show_spinner=False)
def _cmp_hourly_avg(_df: pd.DataFrame):
    return _df.groupby("hour")["cis"].mean().reset_index()


@st.cache_data(show_spinner=False)
def _cmp_vtype_counts(_df: pd.DataFrame):
    vc = _df["vehicle_type"].value_counts().head(8).reset_index()
    vc.columns = ["vehicle_type", "count"]
    return vc


def _cmp_kpi_card(label: str, val_a, val_b, lower_is_better: bool = True):
    if isinstance(val_a, float) and isinstance(val_b, float):
        fmt = lambda v: f"{v:.1f}"
    else:
        fmt = lambda v: f"{int(v):,}"

    delta = val_b - val_a if isinstance(val_a, (int, float)) else 0
    pct   = (delta / val_a * 100) if val_a and val_a != 0 else 0
    good  = (delta < 0) if lower_is_better else (delta > 0)
    dc    = "#00D264" if (good and delta != 0) else "#EF4444" if (not good and delta != 0) else "#888"
    arrow = "▲" if delta > 0 else "▼" if delta < 0 else "—"

    return (
        f"<div style='background:rgba(30,33,48,0.6);border:1px solid rgba(108,99,255,0.15);"
        f"border-radius:12px;padding:14px 16px;text-align:center;'>"
        f"<div style='color:#888;font-size:0.68rem;text-transform:uppercase;"
        f"letter-spacing:.08em;margin-bottom:8px;'>{label}</div>"
        f"<div style='display:flex;justify-content:center;align-items:center;gap:14px;'>"
        f"<div style='color:{_A};font-size:1.05rem;font-weight:800;'>{fmt(val_a)}</div>"
        f"<div style='color:#444;'>→</div>"
        f"<div style='color:{_B};font-size:1.05rem;font-weight:800;'>{fmt(val_b)}</div>"
        f"</div>"
        f"<div style='color:{dc};font-size:0.8rem;font-weight:700;margin-top:5px;'>"
        f"{arrow} {abs(pct):.1f}%</div>"
        f"</div>"
    )


def _cmp_station_compare(stn_a, stn_b, label_a, label_b):
    merged = (
        pd.merge(stn_a, stn_b, on="police_station", suffixes=("_a", "_b"), how="outer")
        .fillna(0)
        .sort_values("violations_a", ascending=True)
        .tail(12)
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(y=merged["police_station"], x=merged["violations_a"],
                         name=label_a, orientation="h",
                         marker=dict(color=_A, opacity=0.85)))
    fig.add_trace(go.Bar(y=merged["police_station"], x=merged["violations_b"],
                         name=label_b, orientation="h",
                         marker=dict(color=_B, opacity=0.85)))
    fig.update_layout(
        barmode="group",
        title=dict(text="Violations by Station", font=dict(size=13, color="#E8E8E8")),
        template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        xaxis=dict(showgrid=False, color="#888"),
        yaxis=dict(showgrid=False, tickfont=dict(size=9, color="#ccc")),
        legend=dict(font=dict(color="#ccc"), bgcolor="rgba(0,0,0,0)"),
        height=370, margin=dict(l=10, r=20, t=40, b=20),
    )
    return fig


def _cmp_hourly_compare(hr_a, hr_b, label_a, label_b):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hr_a["hour"], y=hr_a["cis"], name=label_a,
                             mode="lines+markers",
                             line=dict(color=_A, width=2.5), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=hr_b["hour"], y=hr_b["cis"], name=label_b,
                             mode="lines+markers",
                             line=dict(color=_B, width=2.5), marker=dict(size=5)))
    fig.update_layout(
        title=dict(text="Avg CIS by Hour of Day", font=dict(size=13, color="#E8E8E8")),
        template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        xaxis=dict(title="Hour", tickvals=list(range(0, 24, 2)),
                   gridcolor="rgba(255,255,255,0.05)", color="#888"),
        yaxis=dict(title="Avg CIS", gridcolor="rgba(255,255,255,0.05)", color="#888"),
        legend=dict(font=dict(color="#ccc"), bgcolor="rgba(0,0,0,0)"),
        height=300, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _cmp_vehicle_compare(vt_a, vt_b, label_a, label_b):
    merged = (
        pd.merge(vt_a.rename(columns={"count": "count_a"}),
                 vt_b.rename(columns={"count": "count_b"}),
                 on="vehicle_type", how="outer")
        .fillna(0)
        .sort_values("count_a", ascending=True)
        .tail(8)
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(y=merged["vehicle_type"], x=merged["count_a"],
                         name=label_a, orientation="h",
                         marker=dict(color=_A, opacity=0.85)))
    fig.add_trace(go.Bar(y=merged["vehicle_type"], x=merged["count_b"],
                         name=label_b, orientation="h",
                         marker=dict(color=_B, opacity=0.85)))
    fig.update_layout(
        barmode="group",
        title=dict(text="Violations by Vehicle Type", font=dict(size=13, color="#E8E8E8")),
        template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        xaxis=dict(showgrid=False, color="#888"),
        yaxis=dict(showgrid=False, tickfont=dict(size=9, color="#ccc")),
        legend=dict(font=dict(color="#ccc"), bgcolor="rgba(0,0,0,0)"),
        height=300, margin=dict(l=10, r=20, t=40, b=20),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION RENDERS
# ══════════════════════════════════════════════════════════════════════════════

def _render_enforcement(df):
    st.markdown(
        "<p class='section-header'>Enforcement Priority Command Center</p>"
        "<p class='section-sub'>Stations ranked by Enforcement Priority Index (EPI) "
        "— combining congestion impact, volume, and severity</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='pw-info-banner'>"
        "ℹ️ EPI combines total CIS, violation count, and avg severity — "
        "higher score = more urgent need for enforcement. "
        "Use the table to identify top-priority patrol zones."
        "</div>",
        unsafe_allow_html=True,
    )

    enf = enforcement_table(df)

    # Per-station z-score anomaly detection — flag stations with an internal CIS spike
    _anomaly_names: set[str] = set()
    try:
        from scipy.stats import zscore as _zscore
        for _stn, _vals in df.groupby("police_station", observed=True)["cis"]:
            if len(_vals) >= 2 and float(_zscore(_vals.values).max()) > 2.5:
                _anomaly_names.add(_stn)
    except Exception:
        pass

    top20 = enf.head(20).copy()

    display_df = top20[
        ["police_station", "epi", "violation_count", "avg_cis", "rush_hour_pct"]
    ].copy()
    display_df.columns = ["Station", "EPI Score", "Violations", "Avg CIS", "Rush Hour %"]
    display_df.insert(0, "Rank", range(1, len(display_df) + 1))
    display_df["Avg CIS"]   = display_df["Avg CIS"].round(1)
    display_df["EPI Score"] = display_df["EPI Score"].round(1)

    st.dataframe(
        display_df,
        column_config={
            "Rank":       st.column_config.NumberColumn("Rank", width="small"),
            "Station":    st.column_config.TextColumn("Station", width="medium"),
            "EPI Score":  st.column_config.ProgressColumn("EPI Score", min_value=0, max_value=100, format="%.1f"),
            "Violations": st.column_config.NumberColumn("Violations", format="%d"),
            "Avg CIS":    st.column_config.NumberColumn("Avg CIS", format="%.1f"),
            "Rush Hour %": st.column_config.NumberColumn("Rush Hour %", format="%.1f%%"),
        },
        hide_index=True,
        use_container_width=True,
        height=740,
    )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    top15_enf = enf.head(15)
    n_e = len(top15_enf)
    epi_colors = [
        f"rgba({int(255 - i * 12)}, {int(75 + i * 10)}, {int(75 + i * 8)}, 0.9)"
        for i in range(n_e)
    ]
    fig_epi = go.Figure(
        go.Bar(
            x=top15_enf["epi"].values[::-1],
            y=top15_enf["police_station"].values[::-1],
            orientation="h",
            marker=dict(color=epi_colors[::-1], line=dict(width=0)),
            text=[f"{v:.1f}" for v in top15_enf["epi"].values[::-1]],
            textposition="outside",
            textfont=dict(size=11, color="#ddd"),
        )
    )
    fig_epi.update_layout(
        title=dict(text="Top 15 Stations by Enforcement Priority Index",
                   font=dict(size=15, color="#ddd")),
        xaxis_title="EPI Score",
        yaxis_title=None,
    )
    style_fig(fig_epi, 480)
    st.plotly_chart(fig_epi, use_container_width=True)

    csv_bytes = display_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download Enforcement Table (CSV)", data=csv_bytes,
                       file_name="enforcement_priority.csv", mime="text/csv",
                       key="enf_dl_csv")

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='section-header' style='font-size:1.2rem;'>🚔 Patrol Route Optimizer</p>"
        "<p class='section-sub'>Optimal visit sequence for top-priority stations "
        "using nearest-neighbor routing on actual GPS coordinates</p>",
        unsafe_allow_html=True,
    )

    top_n = st.slider("Number of stations to include in route", 5, 20, 10, key="patrol_n")

    with st.spinner("Computing optimal patrol route…"):
        route_df = _build_patrol_route(df, top_n)

    if route_df.empty:
        st.markdown(
            "<div class='pw-warn-banner'>⚠️ Could not compute route — "
            "latitude/longitude data may be missing.</div>",
            unsafe_allow_html=True,
        )
    else:
        route_disp = route_df[["police_station", "epi", "violation_count", "avg_cis"]].copy()
        route_disp.insert(0, "Stop", [f"#{i+1}" for i in range(len(route_disp))])
        route_disp.columns = ["Stop", "Station", "EPI", "Violations", "Avg CIS"]
        route_disp["EPI"]     = route_disp["EPI"].round(1)
        route_disp["Avg CIS"] = route_disp["Avg CIS"].round(1)
        st.dataframe(route_disp, use_container_width=True, hide_index=True, height=380)

        try:
            import folium
            from streamlit_folium import st_folium

            center_lat = route_df["lat"].mean()
            center_lon = route_df["lon"].mean()
            m = folium.Map(location=[center_lat, center_lon],
                           zoom_start=12, tiles="CartoDB dark_matter")

            coords_list = list(zip(route_df["lat"], route_df["lon"]))
            folium.PolyLine(coords_list, color="#00D2FF", weight=3, opacity=0.7,
                            dash_array="8 4").add_to(m)

            for i, (_, row) in enumerate(route_df.iterrows()):
                icon_html = (
                    f"<div style='background:linear-gradient(135deg,#6C63FF,#00D2FF);"
                    f"color:white;border-radius:50%;width:30px;height:30px;"
                    f"display:flex;align-items:center;justify-content:center;"
                    f"font-weight:800;font-size:12px;"
                    f"box-shadow:0 0 10px rgba(108,99,255,0.7);"
                    f"border:2px solid rgba(255,255,255,0.4);'>{i+1}</div>"
                )
                popup_html = (
                    f"<div style='font-family:Inter,sans-serif;min-width:180px;"
                    f"background:#1a1a2e;border-radius:8px;padding:10px;'>"
                    f"<div style='color:#00D2FF;font-weight:700;font-size:12px;margin-bottom:6px;'>Stop #{i+1}</div>"
                    f"<div style='color:#ccc;font-size:11px;line-height:1.8;'>"
                    f"<b>{row['police_station']}</b><br>"
                    f"EPI: <span style='color:#6C63FF;font-weight:700;'>{row['epi']:.1f}</span><br>"
                    f"Violations: {int(row['violation_count']):,}<br>"
                    f"Avg CIS: {row['avg_cis']:.1f}"
                    f"</div></div>"
                )
                folium.Marker(
                    location=[row["lat"], row["lon"]],
                    icon=folium.DivIcon(html=icon_html, icon_size=(30, 30), icon_anchor=(15, 15)),
                    popup=folium.Popup(popup_html, max_width=220),
                    tooltip=f"Stop #{i+1}: {row['police_station']}",
                ).add_to(m)

            with st.spinner("Rendering patrol map…"):
                st_folium(m, width=None, height=480, returned_objects=[])

        except ImportError:
            st.markdown(
                "<div class='pw-warn-banner'>⚠️ Map view requires folium + streamlit-folium: "
                "<code>pip install folium streamlit-folium</code></div>",
                unsafe_allow_html=True,
            )

        route_csv = route_disp.to_csv(index=False).encode()
        st.download_button("⬇️ Download Patrol Route (CSV)", data=route_csv,
                           file_name="patrol_route.csv", mime="text/csv", key="patrol_dl")

    st.markdown("#### 📐 How is EPI calculated?")
    st.markdown("""
**Congestion Impact Score (CIS)** per violation:
```
CIS = Violation Severity × Vehicle Size × Time Factor × Junction Factor
```
- **Violation Severity**: Double Parking (10) → Wrong Parking (4)
- **Vehicle Size**: Tanker/HGV (10) → Moped (1)
- **Time Factor**: Peak rush hour (3.0) → Late night (0.5), weekends ×0.7
- **Junction Factor**: Near junction (2.0) vs. no junction (1.0)

**Enforcement Priority Index (EPI)** per station:
```
EPI = 0.4 × Norm(Total CIS) + 0.3 × Norm(Violation Count) + 0.3 × Norm(Avg CIS)
```
All scores normalized to 0–100 scale.
    """)

    # ── Section A: Patrol Hours Allocation Board ──────────────────────
    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
    st.subheader("🕐 Patrol Hours Allocation Board")
    st.markdown(
        "<p style='color:#94A3B8;font-size:0.88rem;margin:-8px 0 16px;'>"
        "Daily patrol hours distributed proportionally to EPI — minimum 1 hr per station.</p>",
        unsafe_allow_html=True,
    )

    alloc_df = enf.copy()
    total_epi = alloc_df["epi"].sum()
    alloc_df["patrol_hrs"] = (alloc_df["epi"] / total_epi * 8).clip(lower=1.0).round(1)

    alloc_display = alloc_df[["police_station", "epi", "patrol_hrs", "violation_count"]].copy()
    alloc_display.columns = ["Station", "EPI Score", "Patrol Hours/Day", "Violation Count"]
    alloc_display["EPI Score"] = alloc_display["EPI Score"].round(1)
    st.dataframe(alloc_display, use_container_width=True, hide_index=True, height=400)

    top4 = alloc_df.head(4)
    cols = st.columns(4)
    for col, (_, row) in zip(cols, top4.iterrows()):
        col.metric(
            label=row["police_station"][:22],
            value=f"{row['patrol_hrs']} hrs/day",
            delta=f"EPI {row['epi']:.1f}",
        )

    # ── Section B: Safe Reduction Recommendation ───────────────────────
    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
    st.subheader("📉 Safe Reduction Recommendation")
    st.markdown(
        "<p style='color:#94A3B8;font-size:0.88rem;margin:-8px 0 16px;'>"
        "EPI percentile thresholds determine how safely patrol presence can be reduced per zone.</p>",
        unsafe_allow_html=True,
    )

    p25 = alloc_df["epi"].quantile(0.25)
    p50 = alloc_df["epi"].quantile(0.50)
    p75 = alloc_df["epi"].quantile(0.75)

    def _grade(epi):
        if epi < p25:
            return (50, "✅ Reduce patrol")
        if epi < p50:
            return (30, "⚠️ Minor reduction OK")
        if epi < p75:
            return (10, "🔶 Hold current level")
        return (0, "🚨 Increase enforcement")

    alloc_df[["safe_reduction_pct", "recommendation"]] = alloc_df["epi"].apply(
        lambda e: pd.Series(_grade(e))
    )

    # Override anomaly stations — freeze any reduction recommendation
    if _anomaly_names:
        _amask = alloc_df["police_station"].isin(_anomaly_names)
        alloc_df.loc[_amask, "safe_reduction_pct"] = 0
        alloc_df.loc[_amask, "recommendation"] = "Anomaly detected — freeze reduction"

    # Derive station coordinates and write flagged stations to session state for the map layer
    _stn_coords = (
        df.groupby("police_station", observed=True)
        .agg(lat=("latitude", "median"), lon=("longitude", "median"))
        .reset_index()
    )
    _anomaly_entries = []
    for _s in _anomaly_names:
        _r = _stn_coords[_stn_coords["police_station"] == _s]
        if not _r.empty and pd.notna(_r["lat"].iloc[0]) and pd.notna(_r["lon"].iloc[0]):
            _anomaly_entries.append({
                "name": _s,
                "lat":  float(_r["lat"].iloc[0]),
                "lon":  float(_r["lon"].iloc[0]),
            })
    st.session_state["anomaly_stations"] = _anomaly_entries

    rec_display = alloc_df[["police_station", "epi", "safe_reduction_pct", "recommendation"]].copy()
    rec_display.columns = ["Station", "EPI Score", "Safe Reduction %", "Recommendation"]
    rec_display["EPI Score"] = rec_display["EPI Score"].round(1)
    st.dataframe(rec_display, use_container_width=True, hide_index=True, height=400)

    cat_order  = ["✅ Reduce patrol", "⚠️ Minor reduction OK", "🔶 Hold current level", "🚨 Increase enforcement"]
    cat_colors = ["#10B981", "#F59E0B", "#F97316", "#EF4444"]
    cat_cnt    = alloc_df["recommendation"].value_counts()
    y_vals     = [int(cat_cnt.get(c, 0)) for c in cat_order]

    fig_rec = go.Figure(go.Bar(
        x=cat_order,
        y=y_vals,
        marker=dict(color=cat_colors, line=dict(width=0)),
        text=y_vals,
        textposition="outside",
        textfont=dict(size=12, color="#ddd"),
    ))
    fig_rec.update_layout(
        title=dict(text="Stations by Enforcement Category", font=dict(size=15, color="#ddd")),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        xaxis=dict(tickfont=dict(size=10, color="#aaa")),
        yaxis=dict(title="Stations", gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig_rec, use_container_width=True)

    # ── Section C: Enforcement Effectiveness Score ─────────────────────
    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
    st.subheader("📊 Enforcement Effectiveness Score")
    st.markdown(
        "<p style='color:#94A3B8;font-size:0.88rem;margin:-8px 0 16px;'>"
        "Compares average CIS in the first half of the dataset vs the second half "
        "to measure whether enforcement activity has reduced congestion impact.</p>",
        unsafe_allow_html=True,
    )

    df_eff = df.sort_values("created_datetime").copy()
    mid    = len(df_eff) // 2
    df_bf  = df_eff.iloc[:mid]
    df_af  = df_eff.iloc[mid:]

    mean_cis_before = df_bf["cis"].mean()
    mean_cis_after  = df_af["cis"].mean()

    if mean_cis_before > 0:
        effectiveness = (mean_cis_before - mean_cis_after) / mean_cis_before * 100
    else:
        effectiveness = 0.0

    delta_label = "Congestion reduced" if effectiveness > 0 else "No improvement — review patrol allocation"
    delta_color = "normal" if effectiveness > 0 else "inverse"

    st.metric(
        label="Enforcement Effectiveness",
        value=f"{effectiveness:.1f}%",
        delta=delta_label,
        delta_color=delta_color,
    )

    daily_cis = (
        df_eff
        .assign(date=pd.to_datetime(df_eff["created_datetime"]).dt.date)
        .groupby("date")["cis"]
        .mean()
        .reset_index()
    )
    daily_cis["date_str"] = daily_cis["date"].astype(str)
    mid_idx = len(daily_cis) // 2  # integer index — avoids string arithmetic in add_vline

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=daily_cis["date_str"],
        y=daily_cis["cis"],
        mode="lines+markers",
        line=dict(color="#6C63FF", width=2),
        marker=dict(size=4),
        name="Avg CIS",
        hovertemplate="Date: %{x}<br>Avg CIS: %{y:.1f}<extra></extra>",
    ))
    fig_trend.add_vline(
        x=mid_idx,
        line_dash="dash",
        line_color="#F59E0B",
        annotation_text="Period split",
        annotation_font=dict(color="#F59E0B", size=11),
    )
    fig_trend.update_layout(
        title=dict(text="CIS Trend Over Time (Before vs After Split)", font=dict(size=15, color="#ddd")),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        xaxis=dict(title="Date", tickangle=-30, tickfont=dict(size=9, color="#888"),
                   gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Avg CIS", gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=20, r=20, t=50, b=60),
        showlegend=False,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    before_stn = df_bf.groupby("police_station", observed=True)["cis"].mean().rename("cis_before")
    after_stn  = df_af.groupby("police_station", observed=True)["cis"].mean().rename("cis_after")
    stn_eff    = pd.concat([before_stn, after_stn], axis=1).dropna()
    stn_eff["effectiveness_pct"] = (
        (stn_eff["cis_before"] - stn_eff["cis_after"]) / stn_eff["cis_before"] * 100
    ).round(1)
    stn_eff = stn_eff.sort_values("effectiveness_pct", ascending=False).reset_index()
    stn_eff.columns = ["Station", "CIS Before", "CIS After", "Effectiveness %"]
    stn_eff["CIS Before"] = stn_eff["CIS Before"].round(1)
    stn_eff["CIS After"]  = stn_eff["CIS After"].round(1)

    st.markdown("#### Per-Station Effectiveness")
    st.dataframe(stn_eff, use_container_width=True, hide_index=True)

    # ── Patrol Brief Generator ─────────────────────────────────────────
    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='section-header' style='font-size:1.15rem;'>📋 Patrol Brief Generator</p>"
        "<p class='section-sub'>One-click formal BTP briefing for the duty officer — "
        "powered by Groq LLM or auto-generated from live data</p>",
        unsafe_allow_html=True,
    )

    if st.button("📝 Generate Patrol Brief", key="gen_patrol_brief", use_container_width=True):
        top5 = alloc_df.head(5)[["police_station", "epi", "patrol_hrs"]].copy()
        top5["epi"] = top5["epi"].round(1)
        top5_lines = "\n".join(
            f"  {i+1}. {row['police_station']} — EPI {row['epi']}, {row['patrol_hrs']} hrs/day"
            for i, (_, row) in enumerate(top5.iterrows())
        )
        anomaly_txt = (
            "ANOMALY ALERTS: " + ", ".join(_anomaly_names)
            if _anomaly_names else "No anomalies detected."
        )

        api_key = st.session_state.get("groq_api_key", "")

        if api_key:
            brief_prompt = (
                f"Generate a formal Bengaluru Traffic Police operational patrol briefing "
                f"in under 150 words. Use this data:\n\n"
                f"Top 5 stations by EPI:\n{top5_lines}\n\n"
                f"Enforcement effectiveness: {effectiveness:.1f}%\n"
                f"{anomaly_txt}\n\n"
                f"Format: 3 sections — Situation, Priority Deployment, Alert. "
                f"Professional police tone. No bullet points."
            )
            try:
                from groq import Groq
                _client = Groq(api_key=api_key)
                _resp = _client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": brief_prompt}],
                    max_tokens=300,
                )
                brief_text = _resp.choices[0].message.content
            except Exception as exc:
                brief_text = f"❌ Groq error: {exc}"
        else:
            top1_name = top5.iloc[0]["police_station"] if len(top5) > 0 else "N/A"
            top1_hrs  = top5.iloc[0]["patrol_hrs"]     if len(top5) > 0 else "N/A"
            top2_name = top5.iloc[1]["police_station"] if len(top5) > 1 else "N/A"
            top2_hrs  = top5.iloc[1]["patrol_hrs"]     if len(top5) > 1 else "N/A"
            alert_line = (
                f"Anomaly alerts active at: {', '.join(_anomaly_names)}."
                if _anomaly_names else "All stations within normal parameters."
            )
            brief_text = (
                f"**BTP PATROL OPERATIONS BRIEF**\n\n"
                f"**Situation:** City-wide enforcement effectiveness is at **{effectiveness:.1f}%**. "
                f"{alert_line}\n\n"
                f"**Priority Deployment:** Highest-EPI stations requiring immediate patrol coverage — "
                f"**{top1_name}** ({top1_hrs} hrs/day) and **{top2_name}** ({top2_hrs} hrs/day). "
                f"Ensure tow trucks are pre-positioned at these zones by 0700 hrs. "
                f"Full deployment schedule available in the Patrol Hours Allocation table.\n\n"
                f"**Alert:** All sub-officers to report CIS anomalies above z-score 2.5 to control "
                f"room immediately. EPI rankings updated. Patrol schedules effective 0600 hrs."
            )

        st.markdown(brief_text)
        st.download_button(
            "⬇️ Download Patrol Brief (.md)",
            data=brief_text.encode(),
            file_name="btp_patrol_brief.md",
            mime="text/markdown",
            key="dl_patrol_brief",
        )


def _render_simulator(df: pd.DataFrame) -> None:
    st.markdown(_SIM_CSS, unsafe_allow_html=True)

    st.markdown(
        "<div style='text-align:center;padding:10px 0 26px;'>"
        "<h2 style='background:linear-gradient(90deg,#6C63FF,#00D2FF,#00D264);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        "font-size:2.2rem;font-weight:800;margin-bottom:6px;'>"
        "\U0001f3ae Enforcement Scenario Simulator"
        "</h2>"
        "<p style='color:#888;font-size:1rem;margin:0;'>"
        "Model the impact of targeted enforcement before deploying resources"
        "</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='pw-info-banner'>"
        "ℹ️ <b>How to use:</b> Adjust the sliders to simulate enforcement scenarios — "
        "e.g., reduce violations in Koramangala by 40%. "
        "The projections update instantly to show estimated congestion and revenue impact."
        "</div>",
        unsafe_allow_html=True,
    )

    station_df = compute_epi(df)

    with st.form(key="simulator_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<p style='font-weight:700;color:#ccc;margin-bottom:4px;'>"
                        "\U0001f6e1️ Resources</p>", unsafe_allow_html=True)
            n_teams = st.slider("👮 Patrol Teams Available", 1, 20, 5)
            effectiveness_pct = st.slider("⚡ Enforcement Effectiveness", 20, 80, 50,
                                          help="% reduction in violations per deployed team")

        with col2:
            st.markdown("<p style='font-weight:700;color:#ccc;margin-bottom:4px;'>"
                        "\U0001f3af Strategy</p>", unsafe_allow_html=True)
            strategy   = st.radio("\U0001f3af Deployment Strategy", _STRATEGIES, key="sim_strategy")
            time_focus = st.radio("⏰ Focus Time", _TIME_OPTIONS, key="sim_time")

        with col3:
            st.markdown("<p style='font-weight:700;color:#ccc;margin-bottom:4px;'>"
                        "\U0001f4b0 Economics</p>", unsafe_allow_html=True)
            value_of_time     = st.number_input("\U0001f4b0 Value of Time (₹/hour)",
                                                min_value=100, max_value=500, value=200, step=50)
            vehicles_per_hour = st.number_input("\U0001f697 Avg Vehicles Affected/Hour",
                                                min_value=200, max_value=2000, value=800, step=100)

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("\U0001f680 Run Simulation", use_container_width=True)

    if submitted:
        with st.spinner("\U0001f504 Running simulation…"):
            results = run_simulation(station_df, n_teams, effectiveness_pct, strategy,
                                     time_focus, int(value_of_time), int(vehicles_per_hour))
        st.session_state["sim_results"] = results
        st.session_state["sim_params"]  = {
            "strategy": strategy, "n_teams": n_teams,
            "effectiveness_pct": effectiveness_pct, "time_focus": time_focus,
            "value_of_time": int(value_of_time), "vehicles_per_hour": int(vehicles_per_hour),
        }
        st.session_state["_sim_balloons"] = True

    if st.session_state.pop("_sim_balloons", False):
        st.balloons()

    results = st.session_state.get("sim_results")
    params  = st.session_state.get("sim_params", {})

    if results is None:
        st.markdown(
            "<div style='text-align:center;padding:64px 20px;"
            "background:rgba(30,33,48,0.5);border-radius:16px;"
            "border:1px solid rgba(108,99,255,0.12);margin:20px 0;'>"
            "<div style='font-size:2.5rem;'>\U0001f680</div>"
            "<div style='color:#888;margin-top:12px;font-size:1rem;'>"
            "Configure your scenario above and click "
            "<b style='color:#6C63FF;'>Run Simulation</b>"
            "</div></div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sim-section-hd'>Simulation Results</div>", unsafe_allow_html=True)
    _sim_phase1_metrics(results, station_df)

    st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
    _sim_phase2_before_after(results)

    st.markdown("<div class='sim-section-hd'>Deployment Plan</div>", unsafe_allow_html=True)
    _sim_phase3_table(results)

    st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
    _sim_phase4_strategy_comparison(
        results, station_df,
        params.get("n_teams", 5), params.get("effectiveness_pct", 50),
        params.get("time_focus", "All Day"), params.get("value_of_time", 200),
        params.get("vehicles_per_hour", 800), params.get("strategy", _STRATEGIES[0]),
    )

    st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)
    report = _sim_generate_report(
        results, params.get("strategy", ""), params.get("n_teams", 0),
        params.get("effectiveness_pct", 0), params.get("time_focus", ""),
        params.get("value_of_time", 200),
    )
    st.download_button(label="\U0001f4e5 Download Simulation Report", data=report,
                       file_name="ParkWatch_Simulation.txt", mime="text/plain")


def _render_compare() -> None:
    st.markdown(
        "<p class='section-header'>📊 Before / After Comparison</p>"
        "<p class='section-sub'>Upload two CSVs (e.g., last month vs this month) "
        "and get instant side-by-side delta metrics</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='pw-info-banner'>ℹ️ Upload two separate violation CSVs below. "
        "Both must share the same column schema as the main dataset. "
        "Label them to give each dataset context (e.g., 'January' and 'February').</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"<div style='color:{_A};font-weight:700;font-size:0.9rem;"
                    f"margin-bottom:6px;'>■ Dataset A — Baseline / Before</div>",
                    unsafe_allow_html=True)
        label_a = st.text_input("Label A", value="Before", key="cmp_la")
        file_a  = st.file_uploader("Upload CSV A", type=["csv"], key="cmp_fa")

    with col_b:
        st.markdown(f"<div style='color:{_B};font-weight:700;font-size:0.9rem;"
                    f"margin-bottom:6px;'>■ Dataset B — Comparison / After</div>",
                    unsafe_allow_html=True)
        label_b = st.text_input("Label B", value="After", key="cmp_lb")
        file_b  = st.file_uploader("Upload CSV B", type=["csv"], key="cmp_fb")

    if not file_a or not file_b:
        missing = []
        if not file_a: missing.append("Dataset A")
        if not file_b: missing.append("Dataset B")
        st.markdown(
            f"<div class='pw-empty-state'>"
            f"<div class='es-icon'>📊</div>"
            f"<div class='es-title'>Waiting for {' and '.join(missing)}</div>"
            f"<div class='es-sub'>Upload both CSVs above — "
            f"the comparison will appear here automatically.</div></div>",
            unsafe_allow_html=True,
        )
        return

    with st.spinner("Processing both datasets…"):
        df_a, _ = _load_uploaded(file_a.getvalue(), f"a_{file_a.name}_{file_a.size}")
        df_b, _ = _load_uploaded(file_b.getvalue(), f"b_{file_b.name}_{file_b.size}")

    st.markdown(
        f"<div style='display:flex;gap:12px;margin:16px 0 10px;'>"
        f"<span style='background:{_A}22;border:1px solid {_A}55;border-radius:6px;"
        f"padding:3px 14px;color:{_A};font-size:0.82rem;font-weight:700;'>■ {label_a}</span>"
        f"<span style='background:{_B}22;border:1px solid {_B}55;border-radius:6px;"
        f"padding:3px 14px;color:{_B};font-size:0.82rem;font-weight:700;'>■ {label_b}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        ("Total Violations",  len(df_a),                        len(df_b),                        True),
        ("Avg CIS Score",     float(df_a["cis"].mean()),         float(df_b["cis"].mean()),         True),
        ("High-Risk Count",   int((df_a["cis"] > 50).sum()),     int((df_b["cis"] > 50).sum()),     True),
        ("Stations Affected", df_a["police_station"].nunique(),  df_b["police_station"].nunique(),  False),
    ]
    for col, (label, va, vb, lib) in zip([k1, k2, k3, k4], kpis):
        col.markdown(_cmp_kpi_card(label, va, vb, lib), unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        stn_a = _cmp_top_stations(df_a)
        stn_b = _cmp_top_stations(df_b)
        st.plotly_chart(_cmp_station_compare(stn_a, stn_b, label_a, label_b),
                        use_container_width=True)
    with right:
        hr_a = _cmp_hourly_avg(df_a)
        hr_b = _cmp_hourly_avg(df_b)
        st.plotly_chart(_cmp_hourly_compare(hr_a, hr_b, label_a, label_b),
                        use_container_width=True)

    left2, right2 = st.columns(2)
    with left2:
        vt_a = _cmp_vtype_counts(df_a)
        vt_b = _cmp_vtype_counts(df_b)
        st.plotly_chart(_cmp_vehicle_compare(vt_a, vt_b, label_a, label_b),
                        use_container_width=True)
    with right2:
        rh_a = float(df_a["is_rush_hour"].mean() * 100)
        rh_b = float(df_b["is_rush_hour"].mean() * 100)
        we_a = float(df_a["is_weekend"].mean() * 100)
        we_b = float(df_b["is_weekend"].mean() * 100)
        fig_misc = go.Figure()
        cats   = ["Rush Hour %", "Weekend %", "High CIS %"]
        vals_a = [rh_a, we_a, float((df_a["cis"] > 50).mean() * 100)]
        vals_b = [rh_b, we_b, float((df_b["cis"] > 50).mean() * 100)]
        fig_misc.add_trace(go.Bar(x=cats, y=vals_a, name=label_a,
                                  marker=dict(color=_A, opacity=0.85)))
        fig_misc.add_trace(go.Bar(x=cats, y=vals_b, name=label_b,
                                  marker=dict(color=_B, opacity=0.85)))
        fig_misc.update_layout(
            barmode="group",
            title=dict(text="Pattern Breakdown (%)", font=dict(size=13, color="#E8E8E8")),
            template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
            xaxis=dict(showgrid=False, color="#888"),
            yaxis=dict(title="%", showgrid=False, color="#888"),
            legend=dict(font=dict(color="#ccc"), bgcolor="rgba(0,0,0,0)"),
            height=300, margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_misc, use_container_width=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.markdown("<p class='section-header' style='font-size:1.1rem;'>Station-Level Change Table</p>",
                unsafe_allow_html=True)

    s_a = _cmp_top_stations(df_a, 50)
    s_b = _cmp_top_stations(df_b, 50)
    merged = (
        pd.merge(s_a, s_b, on="police_station", suffixes=("_a", "_b"), how="outer")
        .fillna(0)
    )
    merged["Δ Violations"] = (merged["violations_b"] - merged["violations_a"]).astype(int)
    merged["Δ %"]          = (
        merged["Δ Violations"] / merged["violations_a"].replace(0, np.nan) * 100
    ).round(1).fillna(0)
    merged["Δ CIS"]        = (merged["avg_cis_b"] - merged["avg_cis_a"]).round(1)
    merged = merged.sort_values("Δ Violations")

    disp = merged.rename(columns={
        "police_station": "Station",
        "violations_a":   f"Violations ({label_a})",
        "violations_b":   f"Violations ({label_b})",
        "avg_cis_a":      f"Avg CIS ({label_a})",
        "avg_cis_b":      f"Avg CIS ({label_b})",
    })[[
        "Station",
        f"Violations ({label_a})", f"Violations ({label_b})",
        "Δ Violations", "Δ %",
        f"Avg CIS ({label_a})", f"Avg CIS ({label_b})", "Δ CIS",
    ]]
    st.dataframe(disp, use_container_width=True, hide_index=True)

    csv_bytes = disp.to_csv(index=False).encode()
    st.download_button("⬇️ Download Comparison Table (CSV)", data=csv_bytes,
                       file_name=f"comparison_{label_a}_vs_{label_b}.csv",
                       mime="text/csv", key="cmp_dl")


# ── Preset-card simulator (replaces slider UI in render) ─────────────────────

def _render_simple_simulator(df: pd.DataFrame) -> None:
    st.caption("Pick a preset scenario to see projected impact before deploying resources.")

    scenario = st.radio(
        "Scenario",
        options=[
            "30% fewer violations in top 3 zones",
            "Double peak-hour patrols across all zones",
            "Full enforcement focus on worst zone only",
        ],
        key="preset_scenario",
        label_visibility="collapsed",
    )

    if st.button("▶ Run scenario", type="primary", key="run_preset_btn"):
        station_df = compute_epi(df)
        avg_cis    = float(df["cis"].mean())
        total_viol = max(len(df), 1)

        if scenario == "30% fewer violations in top 3 zones":
            top3_count = int(station_df.head(3)["count"].sum())
            top3_pct   = top3_count / total_viol
            viol_saved = int(top3_count * 0.30)
            cis_delta  = avg_cis * top3_pct * 0.30
            cost_lakh  = viol_saved * 200 * 30 / 1e5
            delay_min  = cis_delta / 100 * 8.0

        elif scenario == "Double peak-hour patrols across all zones":
            rush_col   = "is_rush_hour" if "is_rush_hour" in df.columns else None
            rush_pct   = float(df[rush_col].mean()) if rush_col else 0.35
            viol_saved = int(total_viol * rush_pct * 0.25)
            cis_delta  = avg_cis * rush_pct * 0.25
            cost_lakh  = viol_saved * 200 * 30 / 1e5
            delay_min  = cis_delta / 100 * 8.0

        else:
            worst      = station_df.iloc[0]
            viol_saved = int(worst["count"] * 0.70)
            cis_delta  = float(worst["avg_cis"]) * 0.70 * (worst["count"] / total_viol)
            cost_lakh  = viol_saved * 200 * 30 / 1e5
            delay_min  = cis_delta / 100 * 8.0

        r1, r2, r3 = st.columns(3)
        r1.metric("CIS improvement",   f"−{cis_delta:.1f} pts")
        r2.metric("Cost saved / month", f"₹{cost_lakh:.1f} L")
        r3.metric("Ambulance delay",    f"−{delay_min:.1f} min")


# ── Patrol hours + safe reduction expander content ────────────────────────────

def _render_patrol_hours_expander(df: pd.DataFrame) -> None:
    enf       = enforcement_table(df)
    alloc_df  = enf.copy()
    total_epi = alloc_df["epi"].sum() or 1.0
    alloc_df["patrol_hrs"] = (alloc_df["epi"] / total_epi * 8).clip(lower=1.0).round(1)

    st.caption("Daily patrol hours distributed proportionally to EPI — minimum 1 hr per station.")
    alloc_display = alloc_df[["police_station", "epi", "patrol_hrs", "violation_count"]].copy()
    alloc_display.columns = ["Station", "EPI Score", "Patrol Hours/Day", "Violation Count"]
    alloc_display["EPI Score"] = alloc_display["EPI Score"].round(1)
    st.dataframe(alloc_display, use_container_width=True, hide_index=True, height=400)

    top4 = alloc_df.head(4)
    cols = st.columns(4)
    for col, (_, row) in zip(cols, top4.iterrows()):
        col.metric(label=row["police_station"][:22],
                   value=f"{row['patrol_hrs']} hrs/day",
                   delta=f"EPI {row['epi']:.1f}")

    st.markdown("#### 📉 Safe Reduction Recommendation")
    st.caption("EPI percentile thresholds determine how safely patrol presence can be reduced per zone.")

    p25 = alloc_df["epi"].quantile(0.25)
    p50 = alloc_df["epi"].quantile(0.50)
    p75 = alloc_df["epi"].quantile(0.75)

    def _grade(epi):
        if epi < p25: return (50, "✅ Reduce patrol")
        if epi < p50: return (30, "⚠️ Minor reduction OK")
        if epi < p75: return (10, "🔶 Hold current level")
        return (0, "🚨 Increase enforcement")

    alloc_df[["safe_reduction_pct", "recommendation"]] = alloc_df["epi"].apply(
        lambda e: pd.Series(_grade(e))
    )

    _anomaly_names = {e["name"] for e in st.session_state.get("anomaly_stations", [])}
    if _anomaly_names:
        _amask = alloc_df["police_station"].isin(_anomaly_names)
        alloc_df.loc[_amask, "safe_reduction_pct"] = 0
        alloc_df.loc[_amask, "recommendation"]     = "Anomaly detected — freeze reduction"

    rec_display = alloc_df[["police_station", "epi", "safe_reduction_pct", "recommendation"]].copy()
    rec_display.columns = ["Station", "EPI Score", "Safe Reduction %", "Recommendation"]
    rec_display["EPI Score"] = rec_display["EPI Score"].round(1)
    st.dataframe(rec_display, use_container_width=True, hide_index=True, height=400)

    cat_order  = ["✅ Reduce patrol", "⚠️ Minor reduction OK", "🔶 Hold current level", "🚨 Increase enforcement"]
    cat_colors = ["#10B981", "#F59E0B", "#F97316", "#EF4444"]
    cat_cnt    = alloc_df["recommendation"].value_counts()
    y_vals     = [int(cat_cnt.get(c, 0)) for c in cat_order]

    fig_rec = go.Figure(go.Bar(
        x=cat_order, y=y_vals,
        marker=dict(color=cat_colors, line=dict(width=0)),
        text=y_vals, textposition="outside", textfont=dict(size=12, color="#ddd"),
    ))
    fig_rec.update_layout(
        title=dict(text="Stations by Enforcement Category", font=dict(size=15, color="#ddd")),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        xaxis=dict(tickfont=dict(size=10, color="#aaa")),
        yaxis=dict(title="Stations", gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=20, r=20, t=50, b=20), showlegend=False,
    )
    st.plotly_chart(fig_rec, use_container_width=True)


# ── Enforcement effectiveness expander content ────────────────────────────────

def _render_effectiveness_expander(df: pd.DataFrame) -> None:
    st.caption(
        "Compares average CIS in the first half of the dataset vs the second half "
        "to measure whether enforcement activity has reduced congestion impact."
    )

    df_eff = df.sort_values("created_datetime").copy()
    mid    = len(df_eff) // 2
    df_bf  = df_eff.iloc[:mid]
    df_af  = df_eff.iloc[mid:]

    mean_cis_before = df_bf["cis"].mean()
    mean_cis_after  = df_af["cis"].mean()
    effectiveness   = (
        (mean_cis_before - mean_cis_after) / mean_cis_before * 100
        if mean_cis_before > 0 else 0.0
    )

    delta_label = "Congestion reduced" if effectiveness > 0 else "No improvement — review patrol allocation"
    delta_color = "normal" if effectiveness > 0 else "inverse"
    st.metric(label="Enforcement Effectiveness",
              value=f"{effectiveness:.1f}%",
              delta=delta_label, delta_color=delta_color)

    daily_cis = (
        df_eff
        .assign(date=pd.to_datetime(df_eff["created_datetime"]).dt.date)
        .groupby("date")["cis"].mean().reset_index()
    )
    daily_cis["date_str"] = daily_cis["date"].astype(str)
    mid_idx = len(daily_cis) // 2

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=daily_cis["date_str"], y=daily_cis["cis"],
        mode="lines+markers", line=dict(color="#6C63FF", width=2), marker=dict(size=4),
        name="Avg CIS", hovertemplate="Date: %{x}<br>Avg CIS: %{y:.1f}<extra></extra>",
    ))
    fig_trend.add_vline(
        x=mid_idx, line_dash="dash", line_color="#F59E0B",
        annotation_text="Period split", annotation_font=dict(color="#F59E0B", size=11),
    )
    fig_trend.update_layout(
        title=dict(text="CIS Trend Over Time (Before vs After Split)", font=dict(size=15, color="#ddd")),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        xaxis=dict(title="Date", tickangle=-30, tickfont=dict(size=9, color="#888"),
                   gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Avg CIS", gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=20, r=20, t=50, b=60), showlegend=False,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    before_stn = df_bf.groupby("police_station", observed=True)["cis"].mean().rename("cis_before")
    after_stn  = df_af.groupby("police_station", observed=True)["cis"].mean().rename("cis_after")
    stn_eff    = pd.concat([before_stn, after_stn], axis=1).dropna()
    stn_eff["effectiveness_pct"] = (
        (stn_eff["cis_before"] - stn_eff["cis_after"]) / stn_eff["cis_before"] * 100
    ).round(1)
    stn_eff = stn_eff.sort_values("effectiveness_pct", ascending=False).reset_index()
    stn_eff.columns = ["Station", "CIS Before", "CIS After", "Effectiveness %"]
    stn_eff["CIS Before"] = stn_eff["CIS Before"].round(1)
    stn_eff["CIS After"]  = stn_eff["CIS After"].round(1)

    st.markdown("#### Per-Station Effectiveness")
    st.dataframe(stn_eff, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def render(df):
    # ── FIRST VISIBLE BLOCK: Patrol priority table ────────────────────────────
    st.subheader("Patrol priority — top stations")

    enf = enforcement_table(df)

    alloc_df = enf.copy()
    total_epi = alloc_df["epi"].sum() or 1.0
    alloc_df["patrol_hrs"] = (alloc_df["epi"] / total_epi * 8).clip(lower=1.0).round(1)

    p25, p50, p75 = (
        alloc_df["epi"].quantile(0.25),
        alloc_df["epi"].quantile(0.50),
        alloc_df["epi"].quantile(0.75),
    )

    def _rec(epi: float) -> str:
        if epi < p25: return "✅ Reduce patrol"
        if epi < p50: return "⚠️ Minor reduction OK"
        if epi < p75: return "🔶 Hold current level"
        return "🚨 Increase enforcement"

    alloc_df["recommendation"] = alloc_df["epi"].apply(_rec)
    alloc_df.insert(0, "priority", range(1, len(alloc_df) + 1))

    station_df = alloc_df[
        ["police_station", "epi", "priority", "patrol_hrs", "recommendation"]
    ].copy()
    station_df.columns = ["Station", "EPI Score", "Priority", "Patrol hrs/day", "Recommendation"]
    station_df["EPI Score"] = station_df["EPI Score"].round(1)

    st.dataframe(
        station_df,
        column_config={
            "EPI Score": st.column_config.ProgressColumn(
                "EPI", min_value=0, max_value=float(station_df["EPI Score"].max())
            ),
            "Recommendation": st.column_config.TextColumn("Action"),
        },
        hide_index=True,
        use_container_width=True,
        height=420,
    )

    # ── 4 metric cards for top 4 stations ────────────────────────────────────
    top4 = alloc_df.head(4)
    c1, c2, c3, c4 = st.columns(4)
    for col, (_, row) in zip([c1, c2, c3, c4], top4.iterrows()):
        col.metric(
            label=str(row["police_station"])[:22],
            value=f"{row['patrol_hrs']} hrs/day",
            delta=f"EPI {row['epi']:.1f}",
        )

    # Compute anomaly stations every render so the command-center heatmap layer
    # is always populated regardless of which expanders the user has opened.
    _anomaly_names_set: set[str] = set()
    try:
        from scipy.stats import zscore as _zscore
        for _stn, _vals in df.groupby("police_station", observed=True)["cis"]:
            if len(_vals) >= 2 and float(_zscore(_vals.values).max()) > 2.5:
                _anomaly_names_set.add(_stn)
    except Exception:
        pass
    _stn_coords = (
        df.groupby("police_station", observed=True)
        .agg(lat=("latitude", "median"), lon=("longitude", "median"))
        .reset_index()
    )
    st.session_state["anomaly_stations"] = [
        {"name": _s, "lat": float(_stn_coords.loc[_stn_coords["police_station"] == _s, "lat"].iloc[0]),
                     "lon": float(_stn_coords.loc[_stn_coords["police_station"] == _s, "lon"].iloc[0])}
        for _s in _anomaly_names_set
        if not _stn_coords.loc[_stn_coords["police_station"] == _s].empty
        and pd.notna(_stn_coords.loc[_stn_coords["police_station"] == _s, "lat"].iloc[0])
        and pd.notna(_stn_coords.loc[_stn_coords["police_station"] == _s, "lon"].iloc[0])
    ]

    st.divider()

    with st.expander("🎮 What-if simulator", expanded=False):
        _render_simple_simulator(df)

    with st.expander("🕐 Recommended patrol hours per zone", expanded=False):
        _render_patrol_hours_expander(df)

    with st.expander("📊 Enforcement effectiveness score", expanded=False):
        _render_effectiveness_expander(df)

    with st.expander("🔁 Upload post-enforcement CSV to compare", expanded=False):
        st.caption("Upload two CSVs — baseline and post-enforcement — to calculate improvement metrics.")
        _render_compare()
