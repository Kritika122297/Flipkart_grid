"""
tabs/tab_compare.py  —  Before / After comparison: upload two CSVs, see delta metrics
"""
import io
import os
import tempfile

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from charts.utils import PLOTLY_TEMPLATE, CHART_BG
from data.loader import load_and_process_data

_A = "#6C63FF"   # purple — Dataset A / Before
_B = "#00D264"   # green  — Dataset B / After


# ══════════════════════════════════════════════════════════════════════════════
#  CACHED AGGREGATIONS
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
def _top_stations(_df: pd.DataFrame, n: int = 15):
    return (
        _df.groupby("police_station")
        .agg(violations=("cis", "size"), avg_cis=("cis", "mean"))
        .reset_index()
        .sort_values("violations", ascending=False)
        .head(n)
    )


@st.cache_data(show_spinner=False)
def _hourly_avg(_df: pd.DataFrame):
    return _df.groupby("hour")["cis"].mean().reset_index()


@st.cache_data(show_spinner=False)
def _vtype_counts(_df: pd.DataFrame):
    return _df["vehicle_type"].value_counts().head(8).reset_index()


# ══════════════════════════════════════════════════════════════════════════════
#  HTML HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _kpi_card(label: str, val_a, val_b, lower_is_better: bool = True):
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


# ══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _station_compare(stn_a, stn_b, label_a, label_b):
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


def _hourly_compare(hr_a, hr_b, label_a, label_b):
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


def _vehicle_compare(vt_a, vt_b, label_a, label_b):
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
#  MAIN RENDER
# ══════════════════════════════════════════════════════════════════════════════

def render() -> None:
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

    # ── Uploaders ─────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"<div style='color:{_A};font-weight:700;font-size:0.9rem;"
            f"margin-bottom:6px;'>■ Dataset A — Baseline / Before</div>",
            unsafe_allow_html=True,
        )
        label_a = st.text_input("Label A", value="Before", key="cmp_la")
        file_a  = st.file_uploader("Upload CSV A", type=["csv"], key="cmp_fa")

    with col_b:
        st.markdown(
            f"<div style='color:{_B};font-weight:700;font-size:0.9rem;"
            f"margin-bottom:6px;'>■ Dataset B — Comparison / After</div>",
            unsafe_allow_html=True,
        )
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

    # ── Process both ──────────────────────────────────────────────────────────
    with st.spinner("Processing both datasets…"):
        df_a, _ = _load_uploaded(file_a.getvalue(), f"a_{file_a.name}_{file_a.size}")
        df_b, _ = _load_uploaded(file_b.getvalue(), f"b_{file_b.name}_{file_b.size}")

    # ── Legend chips ──────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='display:flex;gap:12px;margin:16px 0 10px;'>"
        f"<span style='background:{_A}22;border:1px solid {_A}55;border-radius:6px;"
        f"padding:3px 14px;color:{_A};font-size:0.82rem;font-weight:700;'>■ {label_a}</span>"
        f"<span style='background:{_B}22;border:1px solid {_B}55;border-radius:6px;"
        f"padding:3px 14px;color:{_B};font-size:0.82rem;font-weight:700;'>■ {label_b}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── KPI delta cards ───────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        ("Total Violations",      len(df_a),                         len(df_b),                         True),
        ("Avg CIS Score",         float(df_a["cis"].mean()),          float(df_b["cis"].mean()),          True),
        ("High-Risk Count",       int((df_a["cis"] > 50).sum()),      int((df_b["cis"] > 50).sum()),      True),
        ("Stations Affected",     df_a["police_station"].nunique(),    df_b["police_station"].nunique(),   False),
    ]
    for col, (label, va, vb, lib) in zip([k1, k2, k3, k4], kpis):
        col.markdown(_kpi_card(label, va, vb, lib), unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── Charts row 1 ─────────────────────────────────────────────────────────
    left, right = st.columns(2)
    with left:
        stn_a = _top_stations(df_a)
        stn_b = _top_stations(df_b)
        st.plotly_chart(_station_compare(stn_a, stn_b, label_a, label_b),
                        use_container_width=True)
    with right:
        hr_a = _hourly_avg(df_a)
        hr_b = _hourly_avg(df_b)
        st.plotly_chart(_hourly_compare(hr_a, hr_b, label_a, label_b),
                        use_container_width=True)

    # ── Charts row 2 ─────────────────────────────────────────────────────────
    left2, right2 = st.columns(2)
    with left2:
        vt_a = _vtype_counts(df_a)
        vt_b = _vtype_counts(df_b)
        st.plotly_chart(_vehicle_compare(vt_a, vt_b, label_a, label_b),
                        use_container_width=True)
    with right2:
        # Rush hour share comparison
        rh_a = float(df_a["is_rush_hour"].mean() * 100)
        rh_b = float(df_b["is_rush_hour"].mean() * 100)
        we_a = float(df_a["is_weekend"].mean() * 100)
        we_b = float(df_b["is_weekend"].mean() * 100)
        fig_misc = go.Figure()
        cats = ["Rush Hour %", "Weekend %", "High CIS %"]
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

    # ── Station delta table ───────────────────────────────────────────────────
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='section-header' style='font-size:1.1rem;'>Station-Level Change Table</p>",
        unsafe_allow_html=True,
    )

    s_a = _top_stations(df_a, 50)
    s_b = _top_stations(df_b, 50)
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

    # ── Download comparison CSV ───────────────────────────────────────────────
    csv_bytes = disp.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Comparison Table (CSV)",
        data=csv_bytes,
        file_name=f"comparison_{label_a}_vs_{label_b}.csv",
        mime="text/csv",
        key="cmp_dl",
    )
