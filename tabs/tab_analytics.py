import streamlit as st
import plotly.graph_objects as go
from charts.utils import hourly_counts, dow_counts, violation_type_counts, hour_dow_pivot, style_fig


def render(df):
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
                dict(
                    x=8.5, y=hc.max() * 1.1, text="🔴 Rush Hours",
                    showarrow=False, font=dict(color="#FF4B4B", size=10),
                )
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
                marker=dict(
                    color=["#6C63FF"] * 5 + ["#00D2FF"] * 2,
                    line=dict(width=0),
                ),
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
