"""
tabs/tab_cctv.py — 📷 Live CCTV Vision Pipeline
Simulates real-time YOLOv8 edge-AI processing on BTP traffic cameras
to detect illegally parked vehicles without relying on manual CSV tickets.
"""
import streamlit as st
import pandas as pd


def render(df=None):
    st.markdown(
        "<p style='font-size:1.3rem; font-weight:700; color:#818CF8;'>📷 Live CCTV Vision Pipeline</p>"
        "<p style='color:#94A3B8; margin-top:-10px;'>Simulating real-time YOLOv8 edge-processing for illegal parking detection</p>",
        unsafe_allow_html=True,
    )

    st.info(
        "**ℹ️ Concept:** Instead of relying on manual police tickets (CSV data), this module demonstrates how "
        "existing BTP traffic cameras can automatically detect vehicles parked in No-Parking Zones, "
        "compute the live Congestion Impact Score (CIS), and instantly alert the routing dispatcher."
    )

    # --- Build location list from the real dataframe ---
    locations = [
        "Koramangala 80ft Rd",
        "Indiranagar 100ft Rd",
        "Silk Board Junction",
        "MG Road Boulevard",
    ]
    if df is not None:
        for col in ["location", "junction_name", "police_station"]:
            if col in df.columns:
                unique_locs = df[col].dropna().astype(str).unique().tolist()
                if len(unique_locs) > 0:
                    locations = sorted(unique_locs)[:100]
                break

    camera_options = [f"{loc} (Cam {(i % 15) + 1})" for i, loc in enumerate(locations)]

    col1, col2 = st.columns([1, 3])

    with col1:
        camera_sel = st.selectbox("📡 Select Camera Feed", camera_options, key="cctv_camera_sel")
        run_btn = st.button(
            "🔴 Start Live Edge Analysis", type="primary", use_container_width=True, key="cctv_run"
        )
        st.markdown(
            """
            <div style='background:rgba(22,27,47,0.6); padding:14px; border-radius:10px;
                        border:1px solid #334155; margin-top:16px;'>
                <div style='color:#64748B; font-size:0.72rem; text-transform:uppercase;
                            letter-spacing:0.06em;'>Model</div>
                <div style='color:#38BDF8; font-weight:700; margin-bottom:8px;'>YOLOv8-Nano (Edge)</div>
                <div style='color:#64748B; font-size:0.72rem; text-transform:uppercase;
                            letter-spacing:0.06em;'>Compute</div>
                <div style='color:#A78BFA; font-weight:700; margin-bottom:8px;'>NVIDIA Jetson Orin</div>
                <div style='color:#64748B; font-size:0.72rem; text-transform:uppercase;
                            letter-spacing:0.06em;'>Pipeline</div>
                <div style='color:#34D399; font-weight:700;'>Object Det → Tracker → Zone Check</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        if not run_btn:
            st.markdown(
                f"""
                <div style='background:#0f111a; border:2px solid #1E293B; border-radius:12px;
                            padding: 80px 20px; text-align:center; color:#475569;'>
                    <div style='font-size:3.5rem; margin-bottom:16px;'>📷</div>
                    <div style='font-size:1rem; font-weight:600;'>Camera Offline</div>
                    <div style='font-size:0.85rem; margin-top:8px; color:#334155;'>
                        Click <b style='color:#818CF8;'>Start Live Edge Analysis</b>
                        to connect to <b>{camera_sel}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Camera feed simulation using st.markdown (reliable in all environments)
            st.markdown(
                f"""
                <div style='position:relative; background:linear-gradient(135deg,#1a1c29,#0d0f18);
                            border-radius:12px; border:1px solid #4f46e5; padding:16px;
                            box-shadow:0 0 24px rgba(79,70,229,0.25);'>

                    <div style='display:flex; justify-content:space-between; align-items:center;
                                margin-bottom:12px;'>
                        <span style='background:rgba(239,68,68,0.92); color:white; padding:4px 10px;
                                     border-radius:4px; font-size:11px; font-weight:900;
                                     letter-spacing:1px;'>⏺ LIVE REC</span>
                        <span style='background:rgba(0,0,0,0.65); color:#10B981; padding:4px 10px;
                                     border-radius:4px; font-size:11px; font-family:monospace;
                                     border:1px solid #1E293B;'>FPS: 28.4 | {camera_sel}</span>
                    </div>

                    <div style='display:flex; gap:10px; flex-wrap:wrap; min-height:160px;
                                align-items:flex-start; padding:12px 0;'>

                        <div style='flex:1; min-width:160px; background:rgba(239,68,68,0.12);
                                    border:2px solid #EF4444; border-radius:8px; padding:10px;
                                    box-shadow:0 0 12px rgba(239,68,68,0.35);'>
                            <div style='background:#EF4444; color:white; font-size:9px;
                                        font-weight:700; padding:2px 6px; border-radius:2px;
                                        display:inline-block; margin-bottom:8px;'>
                                🚗 Car — 96% | VIOLATION
                            </div>
                            <div style='color:#FCA5A5; font-size:0.75rem;'>
                                Zone: No-Parking Area<br>Action: Alert Dispatched
                            </div>
                        </div>

                        <div style='flex:1; min-width:160px; background:rgba(239,68,68,0.12);
                                    border:2px solid #EF4444; border-radius:8px; padding:10px;
                                    box-shadow:0 0 12px rgba(239,68,68,0.35);'>
                            <div style='background:#EF4444; color:white; font-size:9px;
                                        font-weight:700; padding:2px 6px; border-radius:2px;
                                        display:inline-block; margin-bottom:8px;'>
                                🚛 Truck — 89% | VIOLATION
                            </div>
                            <div style='color:#FCA5A5; font-size:0.75rem;'>
                                Zone: Double Parked<br>Action: Tow Truck Assigned
                            </div>
                        </div>

                        <div style='flex:1; min-width:160px; background:rgba(16,185,129,0.09);
                                    border:2px solid #10B981; border-radius:8px; padding:10px;'>
                            <div style='background:#10B981; color:white; font-size:9px;
                                        font-weight:700; padding:2px 6px; border-radius:2px;
                                        display:inline-block; margin-bottom:8px;'>
                                🚙 Moving — 99% | OK
                            </div>
                            <div style='color:#6EE7B7; font-size:0.75rem;'>
                                Status: In-lane vehicle<br>No action required
                            </div>
                        </div>
                    </div>

                    <div style='color:#334155; font-size:0.7rem; text-align:right; margin-top:8px;'>
                        YOLOv8-Nano · NVIDIA Jetson Orin · Edge Inference
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("### 🚨 Live Infractions Log")
            mock_data = pd.DataFrame({
                "Detection Time":  [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")] * 3,
                "Camera":          [camera_sel] * 3,
                "Vehicle Type":    ["Sedan", "Heavy Goods Truck", "Two-Wheeler"],
                "Confidence":      ["96.4%", "89.1%", "88.2%"],
                "Offense":         ["Parked in No-Parking", "Double Parked (Obstruction)", "Sidewalk Parking"],
                "Est. CIS Impact": [42.5, 88.0, 15.2],
                "Action":          ["Alert Dispatched", "Tow Truck Assigned", "Traffic Fine Issued"],
            })
            st.dataframe(
                mock_data,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Est. CIS Impact": st.column_config.NumberColumn(format="%.1f"),
                    "Action": st.column_config.TextColumn(help="Automated response triggered"),
                },
            )
