import streamlit as st
import plotly.graph_objects as go
from charts.utils import enforcement_table, style_fig


def render(df):
    st.markdown(
        "<p class='section-header'>Enforcement Priority Command Center</p>"
        "<p class='section-sub'>Stations ranked by Enforcement Priority Index (EPI) "
        "— combining congestion impact, volume, and severity</p>",
        unsafe_allow_html=True,
    )

    enf = enforcement_table(df)
    top20 = enf.head(20).copy()

    display_df = top20[
        ["police_station", "epi", "violation_count", "avg_cis", "rush_hour_pct"]
    ].copy()
    display_df.columns = ["Station", "EPI Score", "Violations", "Avg CIS", "Rush Hour %"]
    display_df.insert(0, "Rank", range(1, len(display_df) + 1))
    display_df["Avg CIS"] = display_df["Avg CIS"].round(1)
    display_df["EPI Score"] = display_df["EPI Score"].round(1)

    st.dataframe(
        display_df,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Station": st.column_config.TextColumn("Station", width="medium"),
            "EPI Score": st.column_config.ProgressColumn(
                "EPI Score", min_value=0, max_value=100, format="%.1f"
            ),
            "Violations": st.column_config.NumberColumn("Violations", format="%d"),
            "Avg CIS": st.column_config.NumberColumn("Avg CIS", format="%.1f"),
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
        title=dict(
            text="Top 15 Stations by Enforcement Priority Index",
            font=dict(size=15, color="#ddd"),
        ),
        xaxis_title="EPI Score",
        yaxis_title=None,
    )
    style_fig(fig_epi, 480)
    st.plotly_chart(fig_epi, use_container_width=True)

    with st.expander("📐 **How is EPI calculated?**"):
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
