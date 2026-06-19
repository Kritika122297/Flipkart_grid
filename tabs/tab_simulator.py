"""
Simulator Tab — What-If Enforcement Simulator.
Lets a police commander ask: "If I deploy X patrol teams to Y zones,
how much congestion do I prevent?" — and get projected savings instantly.
"""

from __future__ import annotations
import io
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
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

# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════

_CSS = """<style>
.sim-card {
    background: linear-gradient(135deg, rgba(0,210,100,0.15),
                rgba(0,210,255,0.08));
    backdrop-filter: blur(16px);
    border: 1px solid rgba(0,210,100,0.25);
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 4px;
}
.sim-card .sc-label {
    font-size: 0.78rem; font-weight: 600; color: #888;
    text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 6px;
}
.sim-card .sc-val {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #00D264, #00D2FF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.15;
}
.sim-card .sc-sub {
    font-size: 0.75rem; color: #666; margin-top: 4px;
}
.pro-tip-box {
    background: rgba(108,99,255,0.08);
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 14px;
    padding: 22px 20px;
    backdrop-filter: blur(12px);
    height: 100%;
    box-sizing: border-box;
}
.pro-tip-box .pt-title {
    font-size: 0.88rem; font-weight: 700; color: #00D2FF;
    margin-bottom: 14px; letter-spacing: 0.05em;
}
.pro-tip-box .pt-body {
    font-size: 0.84rem; color: #ccc; line-height: 1.75;
}
.pro-tip-box .pt-hi { color: #00D264; font-weight: 700; }
.sim-section-hd {
    color: #aaa; font-size: 0.77rem; font-weight: 700;
    letter-spacing: .09em; text-transform: uppercase;
    margin: 24px 0 12px;
}
div.stFormSubmitButton > button {
    background: linear-gradient(135deg, #6C63FF, #00D2FF) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px !important;
    transition: opacity 0.2s, box-shadow 0.2s;
}
div.stFormSubmitButton > button:hover {
    opacity: 0.9 !important;
    box-shadow: 0 4px 20px rgba(108,99,255,0.45) !important;
}
</style>"""


# ══════════════════════════════════════════════════════════════════════════════
#  CACHED AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Computing station profiles…")
def compute_epi(_df):
    """
    Aggregate per-station metrics needed for all four deployment strategies.
    Columns required: cis, is_rush_hour, vehicle_size_score, junction_factor,
                      latitude, longitude.
    """
    agg = _df.groupby("police_station").agg(
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


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(
    station_df: pd.DataFrame,
    n_teams: int,
    effectiveness_pct: int,
    strategy: str,
    time_focus: str,
    value_of_time: int,
    vehicles_per_hour: int,
) -> pd.DataFrame:
    """
    Select stations by strategy, then project violation reductions and
    economic savings for the deployment scenario.
    """
    n = min(n_teams, len(station_df))
    sort_col = _STRATEGY_SORT.get(strategy, "epi")
    selected = station_df.sort_values(sort_col, ascending=False).head(n).copy()

    eff    = effectiveness_pct / 100.0
    t_mult = _TIME_MULTIPLIER.get(time_focus, 1.0)

    selected["violations_targeted"] = (selected["count"] * t_mult).astype(int)
    selected["violations_reduced"]  = (selected["violations_targeted"] * eff).astype(int)
    selected["cis_reduced"]         = (selected["avg_cis"] * eff).round(1)

    # avg delay saved per violation × prorated vehicle flow × value-of-time
    avg_delay_min = selected["avg_cis"] / 100.0 * 5.0   # max 5 min delay at CIS=100
    selected["daily_savings_rs"] = (
        selected["violations_reduced"]
        * avg_delay_min
        * (vehicles_per_hour / 60.0)
        * value_of_time
    ).astype(int)
    selected["annual_savings_rs"] = selected["daily_savings_rs"] * 365

    return selected


# ══════════════════════════════════════════════════════════════════════════════
#  RESULT RENDERING — 4 PHASES
# ══════════════════════════════════════════════════════════════════════════════

def _phase1_metrics(results: pd.DataFrame, station_df: pd.DataFrame) -> None:
    cards = [
        ("Violations Prevented",
         f"{results['violations_reduced'].sum():,}",
         "violations prevented daily"),
        ("CIS Reduction",
         f"{results['cis_reduced'].mean():.1f}%",
         "avg congestion impact drop"),
        ("Annual Savings",
         f"₹{results['annual_savings_rs'].sum() / 1e7:.1f} Cr",
         "estimated annual savings"),
        ("Coverage",
         f"{len(results)} / {len(station_df)}",
         "stations covered"),
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


def _phase2_before_after(results: pd.DataFrame) -> None:
    top = results.head(10)
    labels = [
        (s[:28] + "…" if len(s) > 28 else s)
        for s in top["police_station"].tolist()
    ]
    before = top["violations_targeted"].tolist()
    after  = (top["violations_targeted"] - top["violations_reduced"]).tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Before Enforcement",
        x=labels, y=before,
        marker_color="rgba(239,68,68,0.8)",
        marker_line_width=0,
    ))
    fig.add_trace(go.Bar(
        name="After Enforcement",
        x=labels, y=after,
        marker_color="rgba(0,210,100,0.8)",
        marker_line_width=0,
    ))
    fig.update_layout(
        title=dict(
            text="Violation Count: Before vs After Enforcement",
            font=dict(size=15, color="#ddd"),
        ),
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


def _phase3_table(results: pd.DataFrame) -> None:
    display = results[[
        "police_station",
        "violations_targeted",
        "violations_reduced",
        "cis_reduced",
        "daily_savings_rs",
        "annual_savings_rs",
    ]].copy()
    display.insert(0, "Rank", range(1, len(display) + 1))

    st.dataframe(
        display,
        column_config={
            "Rank": st.column_config.NumberColumn(
                "Rank", width="small"
            ),
            "police_station": st.column_config.TextColumn(
                "\U0001f4cd Station", width="medium"
            ),
            "violations_targeted": st.column_config.NumberColumn(
                "Violations Targeted", format="%d"
            ),
            "violations_reduced": st.column_config.NumberColumn(
                "Expected Reduction", format="%d"
            ),
            "cis_reduced": st.column_config.ProgressColumn(
                "CIS Drop", min_value=0, max_value=100, format="%.1f"
            ),
            "daily_savings_rs": st.column_config.NumberColumn(
                "Daily Savings (₹)", format="₹%d"
            ),
            "annual_savings_rs": st.column_config.NumberColumn(
                "Annual Savings (₹)", format="₹%d"
            ),
        },
        use_container_width=True,
        hide_index=True,
        height=420,
    )


def _phase4_strategy_comparison(
    results: pd.DataFrame,
    station_df: pd.DataFrame,
    n_teams: int,
    effectiveness_pct: int,
    time_focus: str,
    value_of_time: int,
    vehicles_per_hour: int,
    strategy: str,
) -> None:
    left, right = st.columns([1.15, 0.85])

    with left:
        st.markdown(
            "<div class='sim-section-hd'>\U0001f4ca Annual Savings by Station</div>",
            unsafe_allow_html=True,
        )
        top8 = results.nlargest(8, "annual_savings_rs")
        labels = [
            (s[:24] + "…" if len(s) > 24 else s)
            for s in top8["police_station"]
        ]
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
            xaxis=dict(showgrid=False, showticklabels=False,
                       title="₹ Lakhs / year"),
            yaxis=dict(showgrid=False, tickfont=dict(size=9, color="#aaa")),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.markdown(
            "<div class='sim-section-hd'>\U0001f4a1 Strategy Intelligence</div>",
            unsafe_allow_html=True,
        )
        total_viol    = station_df["count"].sum()
        sel_coverage  = results["count"].sum()
        sel_pct       = sel_coverage / total_viol * 100 if total_viol else 0.0
        cur_savings   = results["annual_savings_rs"].sum()

        # Always compute EPI baseline for comparison
        epi_sim = run_simulation(
            station_df, n_teams, effectiveness_pct,
            "Top Stations by EPI (Recommended)",
            time_focus, value_of_time, vehicles_per_hour,
        )
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
                f"EPI top zones would cover <span class='pt-hi'>"
                f"{epi_coverage:.1f}%</span> instead.<br><br>"
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


# ══════════════════════════════════════════════════════════════════════════════
#  DOWNLOAD REPORT
# ══════════════════════════════════════════════════════════════════════════════

def _generate_report(
    results: pd.DataFrame,
    strategy: str,
    n_teams: int,
    effectiveness_pct: int,
    time_focus: str,
    value_of_time: int,
) -> str:
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
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def render(df: pd.DataFrame) -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Hero Header ────────────────────────────────────────────────────────────
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

    # Pre-compute station EPI table (cached)
    station_df = compute_epi(df)

    # ── Section 2: Scenario Builder ───────────────────────────────────────────
    with st.form(key="simulator_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                "<p style='font-weight:700;color:#ccc;margin-bottom:4px;'>"
                "\U0001f6e1️ Resources</p>",
                unsafe_allow_html=True,
            )
            n_teams = st.slider("👮 Patrol Teams Available", 1, 20, 5)
            effectiveness_pct = st.slider(
                "⚡ Enforcement Effectiveness", 20, 80, 50,
                help="% reduction in violations per deployed team",
            )

        with col2:
            st.markdown(
                "<p style='font-weight:700;color:#ccc;margin-bottom:4px;'>"
                "\U0001f3af Strategy</p>",
                unsafe_allow_html=True,
            )
            strategy = st.radio(
                "\U0001f3af Deployment Strategy",
                _STRATEGIES,
                key="sim_strategy",
            )
            time_focus = st.radio(
                "⏰ Focus Time",
                _TIME_OPTIONS,
                key="sim_time",
            )

        with col3:
            st.markdown(
                "<p style='font-weight:700;color:#ccc;margin-bottom:4px;'>"
                "\U0001f4b0 Economics</p>",
                unsafe_allow_html=True,
            )
            value_of_time = st.number_input(
                "\U0001f4b0 Value of Time (₹/hour)",
                min_value=100, max_value=500, value=200, step=50,
            )
            vehicles_per_hour = st.number_input(
                "\U0001f697 Avg Vehicles Affected/Hour",
                min_value=200, max_value=2000, value=800, step=100,
            )

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "\U0001f680 Run Simulation", use_container_width=True
        )

    # ── Run & store results ────────────────────────────────────────────────────
    if submitted:
        with st.spinner("\U0001f504 Running simulation…"):
            results = run_simulation(
                station_df,
                n_teams,
                effectiveness_pct,
                strategy,
                time_focus,
                int(value_of_time),
                int(vehicles_per_hour),
            )
        st.session_state["sim_results"] = results
        st.session_state["sim_params"] = {
            "strategy":          strategy,
            "n_teams":           n_teams,
            "effectiveness_pct": effectiveness_pct,
            "time_focus":        time_focus,
            "value_of_time":     int(value_of_time),
            "vehicles_per_hour": int(vehicles_per_hour),
        }
        st.session_state["_sim_balloons"] = True

    # Fire balloons exactly once after each simulation run
    if st.session_state.pop("_sim_balloons", False):
        st.balloons()

    # ── Render results ─────────────────────────────────────────────────────────
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

    # Phase 1 — Headline metrics
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sim-section-hd'>Simulation Results</div>",
        unsafe_allow_html=True,
    )
    _phase1_metrics(results, station_df)

    # Phase 2 — Before vs After chart
    st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
    _phase2_before_after(results)

    # Phase 3 — Deployment plan table
    st.markdown(
        "<div class='sim-section-hd'>Deployment Plan</div>",
        unsafe_allow_html=True,
    )
    _phase3_table(results)

    # Phase 4 — Strategy comparison
    st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
    _phase4_strategy_comparison(
        results,
        station_df,
        params.get("n_teams", 5),
        params.get("effectiveness_pct", 50),
        params.get("time_focus", "All Day"),
        params.get("value_of_time", 200),
        params.get("vehicles_per_hour", 800),
        params.get("strategy", _STRATEGIES[0]),
    )

    # Section 5 — Download report
    st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)
    report = _generate_report(
        results,
        params.get("strategy", ""),
        params.get("n_teams", 0),
        params.get("effectiveness_pct", 0),
        params.get("time_focus", ""),
        params.get("value_of_time", 200),
    )
    st.download_button(
        label="\U0001f4e5 Download Simulation Report",
        data=report,
        file_name="ParkWatch_Simulation.txt",
        mime="text/plain",
    )
