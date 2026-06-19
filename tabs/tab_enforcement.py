import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from charts.utils import enforcement_table, style_fig


# ── Patrol route helper ──────────────────────────────────────────────────────

def _nearest_neighbor(coords: list[tuple]) -> list[int]:
    """Return visit order using nearest-neighbor heuristic on (lat, lon) pairs."""
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
    """Compute top-N EPI stations with coords then nearest-neighbor patrol order."""
    enf = enforcement_table(_df)
    coords_df = (
        _df.groupby("police_station")
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


def render(df):
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

    # ── Download enforcement table ─────────────────────────────────────────────
    csv_bytes = display_df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Enforcement Table (CSV)",
        data=csv_bytes,
        file_name="enforcement_priority.csv",
        mime="text/csv",
        key="enf_dl_csv",
    )

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # ── Patrol Route Optimizer ─────────────────────────────────────────────────
    st.markdown(
        "<p class='section-header' style='font-size:1.2rem;'>🚔 Patrol Route Optimizer</p>"
        "<p class='section-sub'>Optimal visit sequence for top-priority stations "
        "using nearest-neighbor routing on actual GPS coordinates</p>",
        unsafe_allow_html=True,
    )

    top_n = st.slider("Number of stations to include in route", 5, 20, 10,
                       key="patrol_n")

    with st.spinner("Computing optimal patrol route…"):
        route_df = _build_patrol_route(df, top_n)

    if route_df.empty:
        st.markdown(
            "<div class='pw-warn-banner'>⚠️ Could not compute route — "
            "latitude/longitude data may be missing.</div>",
            unsafe_allow_html=True,
        )
    else:
        # ── Route table ───────────────────────────────────────────────────────
        route_disp = route_df[["police_station", "epi", "violation_count", "avg_cis"]].copy()
        route_disp.insert(0, "Stop", [f"#{i+1}" for i in range(len(route_disp))])
        route_disp.columns = ["Stop", "Station", "EPI", "Violations", "Avg CIS"]
        route_disp["EPI"]     = route_disp["EPI"].round(1)
        route_disp["Avg CIS"] = route_disp["Avg CIS"].round(1)
        st.dataframe(route_disp, use_container_width=True, hide_index=True, height=380)

        # ── Folium route map ──────────────────────────────────────────────────
        try:
            import folium
            from streamlit_folium import st_folium

            center_lat = route_df["lat"].mean()
            center_lon = route_df["lon"].mean()
            m = folium.Map(location=[center_lat, center_lon],
                           zoom_start=12, tiles="CartoDB dark_matter")

            # Draw route lines
            coords_list = list(zip(route_df["lat"], route_df["lon"]))
            folium.PolyLine(
                coords_list,
                color="#00D2FF", weight=3, opacity=0.7,
                dash_array="8 4",
            ).add_to(m)

            # Numbered stop markers
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
                    f"<div style='color:#00D2FF;font-weight:700;font-size:12px;"
                    f"margin-bottom:6px;'>Stop #{i+1}</div>"
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

        # ── Download route ────────────────────────────────────────────────────
        route_csv = route_disp.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Patrol Route (CSV)",
            data=route_csv,
            file_name="patrol_route.csv",
            mime="text/csv",
            key="patrol_dl",
        )

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
