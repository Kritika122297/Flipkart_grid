import streamlit as st
import plotly.graph_objects as go
from charts.utils import style_fig


def render(df):
    st.markdown(
        "<p class='section-header'>Economic Impact Calculator</p>"
        "<p class='section-sub'>Quantify the economic cost of parking-induced congestion</p>",
        unsafe_allow_html=True,
    )

    inp1, inp2 = st.columns(2)
    with inp1:
        reduction_pct = st.slider(
            "🔽 Enforcement Reduction %",
            min_value=10,
            max_value=50,
            value=30,
            step=5,
            help="Estimated % reduction in violations through targeted enforcement",
        )
    with inp2:
        time_value = st.number_input(
            "⏱️ Value of Time (₹/hour)",
            min_value=50,
            max_value=1000,
            value=200,
            step=50,
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
        df.groupby("police_station")
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
        title=dict(
            text="Estimated Daily Congestion Cost — Top 15 Stations",
            font=dict(size=15, color="#ddd"),
        ),
        xaxis_title="Daily Cost (₹)",
        yaxis_title=None,
    )
    style_fig(fig_cost, 480)
    st.plotly_chart(fig_cost, use_container_width=True)

    with st.expander("📋 **Assumptions & Methodology**"):
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
