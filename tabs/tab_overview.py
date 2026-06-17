import streamlit as st
import plotly.graph_objects as go
from charts.utils import station_counts, vehicle_counts, ACCENT_SEQ, style_fig


def render(df):
    st.markdown(
        "<p class='section-header'>Dashboard Overview</p>"
        "<p class='section-sub'>Key metrics from ~298K real BTP parking violation records</p>",
        unsafe_allow_html=True,
    )

    total_violations = len(df)
    top_station = df["police_station"].value_counts().idxmax()
    rush_pct = df["is_rush_hour"].mean() * 100
    unique_locs = df["location"].nunique()

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
                <div class='kpi-label'>Top Hotspot Station</div>
                <div class='kpi-value' style='font-size:1.45rem;'>{top_station}</div>
                <div class='kpi-sub'>{df["police_station"].value_counts().iloc[0]:,} violations</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""<div class='kpi-card'>
                <div class='kpi-label'>Rush Hour %</div>
                <div class='kpi-value'>{rush_pct:.1f}%</div>
                <div class='kpi-sub'>7–10 AM & 4–8 PM</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""<div class='kpi-card'>
                <div class='kpi-label'>Unique Locations</div>
                <div class='kpi-value'>{unique_locs:,}</div>
                <div class='kpi-sub'>Distinct addresses</div>
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
