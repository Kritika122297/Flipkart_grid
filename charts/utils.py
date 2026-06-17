import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from collections import Counter

GRADIENT_COLORS = [
    "#6C63FF", "#7C6FFF", "#8B7BFF", "#9B87FF", "#AA93FF",
    "#B9A0FF", "#C5ADFF", "#00D2FF", "#00BFFF", "#00AAFF",
    "#0095FF", "#0080FF", "#006BFF", "#0056FF", "#0041FF",
]
ACCENT_SEQ = ["#6C63FF", "#00D2FF", "#7C4DFF", "#FF6B6B", "#FFD93D",
              "#6BCB77", "#4D96FF", "#FF6B9D", "#C084FC", "#22D3EE"]
PLOTLY_TEMPLATE = "plotly_dark"
CHART_BG = "rgba(0,0,0,0)"


def style_fig(fig, height=420):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(family="Inter", color="#bbb"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=height,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#aaa"),
        ),
    )
    return fig


@st.cache_data
def station_counts(_df):
    return _df["police_station"].value_counts().head(15)


@st.cache_data
def vehicle_counts(_df):
    return _df["vehicle_type"].value_counts().head(10)


@st.cache_data
def hourly_counts(_df):
    return _df.groupby("hour").size().reindex(range(24), fill_value=0)


@st.cache_data
def dow_counts(_df):
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    vc = _df["day_of_week"].value_counts()
    return vc.reindex(order).fillna(0).astype(int)


@st.cache_data
def violation_type_counts(_df):
    cnt = Counter()
    for vlist in _df["violation_list"]:
        for v in vlist:
            cnt[v] += 1
    return pd.Series(cnt).sort_values(ascending=False).head(15)


@st.cache_data
def hour_dow_pivot(_df):
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = _df.groupby(["hour", "day_of_week"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=order, fill_value=0).reindex(range(24), fill_value=0)
    return pivot


@st.cache_data
def enforcement_table(_df):
    agg = _df.groupby("police_station").agg(
        total_cis=("cis", "sum"),
        avg_cis=("cis", "mean"),
        violation_count=("id", "size"),
        rush_hour_pct=("is_rush_hour", "mean"),
    ).reset_index()
    agg["rush_hour_pct"] = (agg["rush_hour_pct"] * 100).round(1)

    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) * 100 if mx > mn else 50

    agg["norm_total_cis"] = norm(agg["total_cis"])
    agg["norm_count"] = norm(agg["violation_count"])
    agg["norm_avg_cis"] = norm(agg["avg_cis"] * 100)
    agg["epi"] = (
        0.4 * agg["norm_total_cis"]
        + 0.3 * agg["norm_count"]
        + 0.3 * agg["norm_avg_cis"]
    )
    epi_min, epi_max = agg["epi"].min(), agg["epi"].max()
    agg["epi"] = (
        ((agg["epi"] - epi_min) / (epi_max - epi_min) * 100)
        if epi_max > epi_min
        else 50
    )
    agg = agg.sort_values("epi", ascending=False).reset_index(drop=True)
    agg.index += 1
    return agg
