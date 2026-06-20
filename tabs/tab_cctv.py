"""
tabs/tab_cctv.py — 📷 Live CCTV Vision Pipeline
Added by: Vatsalya (vatsalyadwiv1111) — Rank 1 Feature
Simulates real-time YOLOv8 edge-AI processing on BTP traffic cameras
to detect illegally parked vehicles without relying on manual CSV tickets.
"""
import streamlit as st
import pandas as pd
import time


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

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

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

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style='background:rgba(22,27,47,0.6); padding:14px; border-radius:10px; border:1px solid #334155;'>
                <div style='color:#64748B; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em;'>Model</div>
                <div style='color:#38BDF8; font-weight:700; margin-bottom:8px;'>YOLOv8-Nano (Edge)</div>
                <div style='color:#64748B; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em;'>Compute</div>
                <div style='color:#A78BFA; font-weight:700; margin-bottom:8px;'>NVIDIA Jetson Orin</div>
                <div style='color:#64748B; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em;'>Pipeline</div>
                <div style='color:#34D399; font-weight:700;'>Object Det → Tracker → Zone Check</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        if not run_btn:
            st.html(
                f"<div style='background:#0f111a; border:1px solid #1E293B; border-radius:12px; height:420px;"
                f"display:flex; align-items:center; justify-content:center; color:#475569; flex-direction:column;'>"
                f"<div style='font-size:3rem; margin-bottom:10px;'>📷</div>"
                f"<div>Camera Offline. Click 'Start' to connect to {camera_sel}.</div></div>"
            )
        else:
            with st.spinner(f"Connecting to {camera_sel} and initializing YOLOv8-Nano weights..."):
                time.sleep(2)

            st.html(f"""
<div style='position:relative; width:100%; height:420px;
     background:linear-gradient(135deg, #1a1c29, #0d0f18);
     border-radius:12px; border:1px solid #4f46e5; overflow:hidden;
     box-shadow: 0 0 24px rgba(79,70,229,0.25);'>

  <div style='position:absolute; top:10px; left:10px;
       background:rgba(239,68,68,0.92); color:white; padding:4px 10px;
       border-radius:4px; font-size:11px; font-weight:900; letter-spacing:1px;
       display:flex; align-items:center; gap:6px;'>
    <span style='display:inline-block; width:8px; height:8px; background:white;
          border-radius:50%; animation:blink 1s infinite;'></span>
    LIVE REC
  </div>

  <div style='position:absolute; top:10px; right:10px;
       background:rgba(0,0,0,0.65); padding:4px 10px; border-radius:4px;
       color:#10B981; font-family:monospace; font-size:11px; border:1px solid #1E293B;'>
    FPS: 28.4 | {camera_sel}
  </div>

  <!-- Road lane markings -->
  <div style='position:absolute; top:0; bottom:0; left:30%; width:3px;
       background:rgba(255,255,255,0.07); border-right:2px dashed rgba(255,255,255,0.2);'></div>
  <div style='position:absolute; top:0; bottom:0; left:70%; width:3px;
       background:rgba(255,255,255,0.07); border-right:2px dashed rgba(255,255,255,0.2);'></div>

  <!-- No Parking Zone -->
  <div style='position:absolute; bottom:8%; left:4%; width:22%; height:82%;
       background:rgba(239,68,68,0.05); border:2px dashed rgba(239,68,68,0.45);
       display:flex; justify-content:center; align-items:flex-start; padding-top:18px;
       color:rgba(239,68,68,0.65); font-weight:900; font-size:12px;
       transform:skewX(-8deg); letter-spacing:0.05em;'>NO PARKING ZONE</div>

  <!-- Violation: Car -->
  <div style='position:absolute; bottom:140px; left:7%; width:140px; height:68px;
       border:2px solid #EF4444; background:rgba(239,68,68,0.14);
       display:flex; align-items:flex-start; padding:3px;
       box-shadow:0 0 12px rgba(239,68,68,0.45);'>
    <span style='background:#EF4444; color:white; font-size:9px; font-weight:700;
          padding:2px 6px; border-radius:2px;'>Car 96% | VIOLATION</span>
  </div>

  <!-- Violation: Truck -->
  <div style='position:absolute; bottom:18px; left:11%; width:160px; height:78px;
       border:2px solid #EF4444; background:rgba(239,68,68,0.14);
       display:flex; align-items:flex-start; padding:3px;
       box-shadow:0 0 12px rgba(239,68,68,0.45);'>
    <span style='background:#EF4444; color:white; font-size:9px; font-weight:700;
          padding:2px 6px; border-radius:2px;'>Truck 89% | VIOLATION</span>
  </div>

  <!-- OK: Moving vehicle -->
  <div style='position:absolute; top:90px; left:44%; width:105px; height:58px;
       border:2px solid #10B981; background:rgba(16,185,129,0.09);
       display:flex; align-items:flex-start; padding:3px;'>
    <span style='background:#10B981; color:white; font-size:9px; font-weight:700;
          padding:2px 6px; border-radius:2px;'>Moving 99%</span>
  </div>

  <style>
  @keyframes blink {{
    0%   {{ opacity:1; }}
    50%  {{ opacity:0; }}
    100% {{ opacity:1; }}
  }}
  </style>
</div>
""")

            st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

            mock_data = pd.DataFrame({
                "Detection Time":  [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")] * 3,
                "Camera":          [camera_sel] * 3,
                "Vehicle Type":    ["Sedan", "Heavy Goods Truck", "Two-Wheeler"],
                "Confidence":      ["96.4%", "89.1%", "88.2%"],
                "Offense":         ["Parked in No-Parking", "Double Parked (Obstruction)", "Sidewalk Parking"],
                "Est. CIS Impact": [42.5, 88.0, 15.2],
                "Action":          ["Alert Dispatched", "Tow Truck Assigned", "Traffic Fine Issued"],
            })

            st.markdown("### 🚨 Live Infractions Log")
            st.dataframe(
                mock_data,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Est. CIS Impact": st.column_config.NumberColumn(format="%.1f"),
                    "Action": st.column_config.TextColumn(help="Automated response triggered"),
                },
            )
