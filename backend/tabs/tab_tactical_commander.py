"""
tabs/tab_tactical_commander.py — 🤖 Tactical AI Commander
Sections: A) Gemini AI Chat  B) Dispatch Planner  C) RF Risk Forecaster (via tab_tactical_ai)

Public API:
  render_chat_panel(df)  — floating-ready chat; call from app.py (any tab context)
  render(df)             — Tab 4 body: dispatch planner + RF forecaster only
"""
import time
import streamlit as st
import pandas as pd

from tabs import tab_tactical_ai


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO MODE — hardcoded Q&A for presentations without an API key
# ══════════════════════════════════════════════════════════════════════════════

_DEMO_QA = {
    "Where should we deploy tow trucks in Koramangala?": (
        "Based on CIS data for **Koramangala**, deploy **3 tow trucks** at:\n\n"
        "1. **80 Feet Road junction** (7–10 AM, 4–8 PM) — highest double-parking density "
        "near commercial strips.\n"
        "2. **Sarjapur Road & 5th Block** — peak CIS of 78 during evening rush. "
        "Stage at Jyoti Nivas College junction for fastest response.\n"
        "3. **100 Feet Road** — afternoon commercial overflow from the market. "
        "Rotate with 80 Feet Road unit post 10 AM.\n\n"
        "**Estimated clearance time:** 12–18 min per incident. "
        "Coordinate with HSR Layout station for weekend overflow support."
    ),
    "Why is there a traffic spike near Silk Board?": (
        "The Silk Board CIS spike has three compounding causes:\n\n"
        "1. **Junction factor ×2.0** — classified as a major junction, doubling CIS "
        "for every violation within 50 m.\n"
        "2. **Heavy vehicles** — Tankers and HGVs from the Electronic City industrial "
        "corridor park on slip roads during delivery windows (9–11 AM, 2–4 PM), "
        "each scoring ×10 on the vehicle-size multiplier.\n"
        "3. **Time factor ×3.0** — violations concentrated during peak rush amplify "
        "base CIS threefold.\n\n"
        "**Fix:** Station one officer at the Silk Board underpass ramp 7–10 AM. "
        "Request BBMP no-parking boards on Hosur Road service lane between 8–11 AM."
    ),
    "Generate an executive summary for the commissioner": (
        "**ParkWatch AI — Executive Briefing**\n\n"
        "Parking-induced congestion accounts for an estimated **₹2.4 Cr/day** in "
        "economic loss across Bengaluru, based on CIS-weighted vehicle delay modelling.\n\n"
        "**Key findings:**\n"
        "- Top 5 enforcement zones → 42% of total CIS load\n"
        "- Morning rush (7–10 AM) → 3× congestion impact vs off-peak hours\n"
        "- Heavy vehicles (tankers, HGVs) → 6.8× disproportionate CIS per violation\n"
        "- Top-10 EPI station enforcement → projected 28% city-wide CIS reduction\n\n"
        "**Recommended action:** Authorise a 6-week pilot deploying 8 additional tow trucks "
        "to top-10 EPI stations during morning rush. "
        "Projected ROI: ₹12 Cr in prevented economic loss vs ₹18 L operational cost."
    ),
}

_DEMO_DISPATCH = (
    "**5-Step Tactical Dispatch Plan**\n\n"
    "1. **Activate 2 tow trucks** — Stage at the station's primary junction approach, "
    "one on each side of the road to handle both directions simultaneously.\n\n"
    "2. **Officer positioning** — Place 1 officer 200 m upstream to divert traffic and "
    "1 at the hotspot to coordinate tow operations. Use radio for real-time sync.\n\n"
    "3. **Diversion route** — Log incident on Google Maps & BBMP traffic system. "
    "Request signal control to extend green phases on the parallel alternate route.\n\n"
    "4. **Tow sequence** — Tankers and HGVs first (highest CIS impact), "
    "then double-parked cars. Photograph and challan before towing for e-challan upload.\n\n"
    "5. **Estimated clearance** — 15–22 minutes for full clearance. "
    "Report to dispatch on WhatsApp group once zone is clear. "
    "Stand down at 30 min if no new violations detected."
)


# ══════════════════════════════════════════════════════════════════════════════
#  GEMINI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_system_prompt(df: pd.DataFrame) -> str:
    top3 = df.groupby("police_station", observed=True)["cis"].mean().nlargest(3)
    peak_hour = int(df.groupby("hour")["cis"].mean().idxmax())
    total_violations = len(df)
    total_stations = df["police_station"].nunique()
    avg_cis = float(df["cis"].mean())
    top3_lines = "\n".join(f"  - {stn}: avg CIS {cis:.1f}" for stn, cis in top3.items())

    return (
        "You are an expert AI tactical advisor for the Bengaluru Traffic Police (BTP), "
        "specialised in parking-induced congestion management. "
        "You have access to live violation data. Current dataset summary:\n\n"
        f"- Total violations analysed: {total_violations:,}\n"
        f"- Unique enforcement stations: {total_stations}\n"
        f"- City-wide average CIS (Congestion Impact Score): {avg_cis:.1f}\n"
        f"- Peak congestion hour: {peak_hour:02d}:00\n"
        f"- Top 3 high-impact stations:\n{top3_lines}\n\n"
        "CIS = Violation Severity × Vehicle Size × Time Factor × Junction Factor\n"
        "EPI = 0.4×Norm(Total CIS) + 0.3×Norm(Count) + 0.3×Norm(Avg CIS), scaled 0–100\n\n"
        "Always give specific, actionable advice referencing real Bengaluru roads and areas. "
        "Keep responses concise with clear headings and bullet points."
    )


def _call_groq(api_key: str, system_prompt: str, user_msg: str, history: list) -> str:
    try:
        from groq import Groq
    except ImportError:
        return (
            "❌ `groq` not installed.\n\n"
            "Run: `pip install groq` then reload the app."
        )
    try:
        client = Groq(api_key=api_key)
        messages = [{"role": "system", "content": system_prompt}]
        for m in history:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": user_msg})
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=1024,
        )
        return resp.choices[0].message.content
    except Exception as exc:
        return f"❌ Groq error: {exc}"


def _cached_call(api_key: str, system_prompt: str, user_msg: str, history: list) -> str:
    if "groq_cache" not in st.session_state:
        st.session_state["groq_cache"] = {}

    cache_key = user_msg.strip().lower()[:300]
    entry = st.session_state["groq_cache"].get(cache_key)
    if entry and time.time() - entry[0] < 60:
        return entry[1]

    response = _call_groq(api_key, system_prompt, user_msg, history)
    st.session_state["groq_cache"][cache_key] = (time.time(), response)
    return response


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION RENDERS
# ══════════════════════════════════════════════════════════════════════════════

def render_chat_panel(df: pd.DataFrame) -> None:
    """Width-agnostic chat panel — safe to call from any tab context or app.py."""
    api_key: str = st.session_state.get("groq_api_key", "")
    system_prompt = _build_system_prompt(df)

    st.markdown(
        "<p class='section-header'>🧠 AI Chat Assistant</p>"
        "<p class='section-sub'>Ask anything about your violation data — patrol deployment, "
        "CIS spikes, executive summaries, diversion strategies</p>",
        unsafe_allow_html=True,
    )

    # ── Demo mode ─────────────────────────────────────────────────────────────
    if not api_key:
        st.warning(
            "Groq API key not configured in the environment — showing demo responses."
        )
        sample_q = st.selectbox(
            "Try a sample question:",
            options=list(_DEMO_QA.keys()),
            key="demo_q_sel",
        )
        with st.chat_message("user"):
            st.markdown(sample_q)
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(_DEMO_QA[sample_q])
        return

    # ── Live Groq chat ─────────────────────────────────────────────────────────
    if st.button("🗑️ Clear chat history", key="clear_chat"):
        st.session_state["chat_history"] = []
        st.rerun()

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else None):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about Bengaluru traffic data…", key="chat_input_panel"):
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        history_before = st.session_state["chat_history"][:-1]
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Groq is thinking…"):
                response = _cached_call(api_key, system_prompt, prompt, history_before)
            st.markdown(response)

        st.session_state["chat_history"].append({"role": "assistant", "content": response})


def _render_dispatch(df: pd.DataFrame, api_key: str, system_prompt: str) -> None:
    st.markdown(
        "<p class='section-header' style='font-size:1.2rem;'>🚔 Dispatch Planner</p>"
        "<p class='section-sub'>Generate an AI-powered 5-step tactical dispatch and "
        "diversion plan for any enforcement station</p>",
        unsafe_allow_html=True,
    )

    stations = sorted(df["police_station"].dropna().unique().tolist())
    vehicle_types = (
        sorted(df["vehicle_type"].dropna().unique().tolist())
        if "vehicle_type" in df.columns
        else ["Car", "Two-wheeler", "Bus", "HGV/Truck", "Tanker"]
    )

    col1, col2 = st.columns(2)
    with col1:
        selected_station = st.selectbox(
            "Generate dispatch plan for:", stations, key="dispatch_station"
        )
    with col2:
        selected_vtype = st.selectbox(
            "Primary violation type:", vehicle_types, key="dispatch_vtype"
        )

    avg_cis = float(df[df["police_station"] == selected_station]["cis"].mean())

    st.markdown(
        f"<div style='color:#666;font-size:0.78rem;margin:4px 0 12px;'>"
        f"Station avg CIS: <b style='color:#F59E0B;'>{avg_cis:.1f}</b></div>",
        unsafe_allow_html=True,
    )

    if st.button("🎯 Generate Dispatch Plan", key="gen_dispatch", use_container_width=True):
        if not api_key:
            st.markdown(
                "<div class='pw-info-banner'>ℹ️ Demo mode — showing sample dispatch plan</div>",
                unsafe_allow_html=True,
            )
            st.markdown(_DEMO_DISPATCH)
        else:
            dispatch_prompt = (
                f"You are a BTP tactical officer. Generate a numbered 5-step dispatch and "
                f"traffic diversion plan for {selected_station} in Bengaluru, where the main "
                f"issue is {selected_vtype} violations causing {avg_cis:.1f} average congestion "
                f"impact. Be specific to Bengaluru roads. Include: tow truck count, staging "
                f"location, diversion route, officer positions, estimated clearance time."
            )
            with st.spinner("Generating dispatch plan…"):
                plan = _cached_call(api_key, system_prompt, dispatch_prompt, [])
            st.markdown(plan)


def _render_rf(df: pd.DataFrame) -> None:
    st.subheader("Congestion Risk Forecaster (ML)")
    tab_tactical_ai.render(df)


# ── Data-driven demo Q&A ──────────────────────────────────────────────────────

def _build_demo_qa(df: pd.DataFrame) -> dict:
    top3     = df.groupby("police_station", observed=True)["cis"].mean().nlargest(3)
    top1     = top3.index[0]    if len(top3) > 0 else "Koramangala"
    cis1     = round(top3.iloc[0], 1) if len(top3) > 0 else 8.4
    top2     = top3.index[1]    if len(top3) > 1 else "Indiranagar"
    cis2     = round(top3.iloc[1], 1) if len(top3) > 1 else 7.1
    peak_hr  = int(df.groupby("hour")["cis"].mean().idxmax())
    avg_cis  = round(float(df["cis"].mean()), 1)
    spike_x  = round(cis1 / max(avg_cis, 0.1), 1)
    peak_end = (peak_hr + 2) % 24
    anomaly_count = len(st.session_state.get("anomaly_stations", [])) or 3
    top_vtype = "Two-wheelers"
    if "vehicle_type" in df.columns:
        s_df = df[df["police_station"] == top1]
        if len(s_df) > 0:
            top_vtype = s_df["vehicle_type"].value_counts().index[0]
    monthly_savings_l = round(
        max(avg_cis * len(df) * 200 * 30 / 1e7 * 100 * 0.28, 1.0), 1
    )

    return {
        f"Where should I deploy tow trucks at {top1} today?": (
            f"Based on current CIS data, **{top1}** is the highest-risk zone "
            f"(avg CIS {cis1} — {spike_x}× city average).\n\n"
            f"**Deploy 2–3 tow trucks:**\n"
            f"- **{peak_hr:02d}:00–{peak_end:02d}:00** — stage at the primary "
            f"junction approach to intercept inbound peak-hour violators.\n"
            f"- **17:00–20:00** — rotate to commercial side-roads where "
            f"double-parking density peaks in the evening.\n\n"
            f"**Primary offender:** {top_vtype} — prioritise these for fastest "
            f"flow restoration. Estimated clearance per incident: 12–18 min."
        ),
        f"Why is {top2} spiking right now?": (
            f"**{top2}** shows avg CIS {cis2} vs city avg {avg_cis} "
            f"— a **{round(cis2 / max(avg_cis, 0.1), 1)}× spike**. Primary causes:\n\n"
            f"- **Junction multiplier ×2.0** — classified major junction, doubling CIS "
            f"for every violation within 50 m.\n"
            f"- **Peak-hour concentration** — {peak_hr:02d}:00 window carries "
            f"3× off-peak load.\n"
            f"- **Heavy vehicle mix** — tankers/HGVs score ×10 on the "
            f"vehicle-size multiplier.\n\n"
            f"**Recommend:** Deploy 1 officer at the primary merge point immediately. "
            f"Request signal timing adjustment from BBMP traffic control."
        ),
        "Generate an executive summary for the BTP Commissioner": (
            f"**BTP Enforcement Brief — Today**\n\n"
            f"Top concern: **{top1}** (CIS {cis1}), **{top2}** (CIS {cis2}). "
            f"Peak window: {peak_hr:02d}:00–{peak_end:02d}:00. "
            f"{anomaly_count} anomalous zones detected.\n\n"
            f"Recommended action: increase **{top1}** patrol by 2 officers during AM peak. "
            f"Estimated monthly savings if enforced: ₹{monthly_savings_l} L."
        ),
    }


# ── Executive briefing expander content ──────────────────────────────────────

def _render_exec_briefing(
    df: pd.DataFrame, api_key: str, system_prompt: str
) -> None:
    st.caption(
        "One-click formal briefing for the BTP Commissioner — "
        "generated from live enforcement data."
    )

    if st.button("📄 Generate Briefing", key="gen_exec_brief", use_container_width=True):
        top3    = df.groupby("police_station", observed=True)["cis"].mean().nlargest(3)
        top1    = top3.index[0] if len(top3) > 0 else "N/A"
        cis1    = round(top3.iloc[0], 1) if len(top3) > 0 else 0.0
        top2    = top3.index[1] if len(top3) > 1 else "N/A"
        cis2    = round(top3.iloc[1], 1) if len(top3) > 1 else 0.0
        peak_hr = int(df.groupby("hour")["cis"].mean().idxmax())
        avg_cis = round(float(df["cis"].mean()), 1)
        anomaly_count = len(st.session_state.get("anomaly_stations", []))
        monthly_savings_l = round(
            max(avg_cis * len(df) * 200 * 30 / 1e7 * 100 * 0.28, 1.0), 1
        )
        peak_end = (peak_hr + 2) % 24

        if not api_key:
            brief_text = (
                f"**BTP ENFORCEMENT BRIEF — PARKWATCH AI**\n\n"
                f"**SITUATION:** City-wide avg CIS: {avg_cis}. "
                f"Peak congestion at {peak_hr:02d}:00. "
                f"{anomaly_count} anomalous zones flagged by AI model.\n\n"
                f"**PRIORITY DEPLOYMENT:** {top1} (CIS {cis1}) and "
                f"{top2} (CIS {cis2}) require immediate reinforcement. "
                f"Deploy additional units during "
                f"{peak_hr:02d}:00–{peak_end:02d}:00 window.\n\n"
                f"**ALERT:** Enforce zero-tolerance on heavy vehicle violations. "
                f"Estimated monthly savings with full enforcement: ₹{monthly_savings_l} L. "
                f"EPI rankings updated — patrol schedules effective 0600 hrs."
            )
        else:
            top3_lines = "\n".join(
                f"  - {s}: avg CIS {c:.1f}" for s, c in top3.items()
            )
            brief_prompt = (
                f"Generate a formal Bengaluru Traffic Police operational briefing "
                f"for the BTP Commissioner in under 150 words. Data:\n\n"
                f"Top 3 stations by CIS:\n{top3_lines}\n"
                f"Peak hour: {peak_hr:02d}:00\n"
                f"City avg CIS: {avg_cis}\n"
                f"Anomalous zones: {anomaly_count}\n\n"
                f"Format: 3 sections — Situation, Priority Deployment, Alert. "
                f"Professional police tone. No bullet points."
            )
            with st.spinner("Generating briefing…"):
                brief_text = _cached_call(api_key, system_prompt, brief_prompt, [])

        st.markdown(brief_text)
        st.download_button(
            "⬇️ Download Briefing (.md)",
            data=brief_text.encode(),
            file_name="btp_commissioner_brief.md",
            mime="text/markdown",
            key="dl_exec_brief",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def render(df: pd.DataFrame) -> None:
    api_key: str  = st.session_state.get("groq_api_key", "")
    system_prompt = _build_system_prompt(df)

    # ── 1. ALWAYS VISIBLE: AI chat or demo ────────────────────────────────────
    st.subheader("🧠 AI Tactical Advisor")

    if not api_key:
        st.warning("Demo mode — no Groq API key. Select a sample question below.")
        demo_qa    = _build_demo_qa(df)
        selected_q = st.selectbox(
            "Try a sample question:", list(demo_qa.keys()), key="demo_q_render"
        )
        st.markdown("**AI Response:**")
        st.info(demo_qa[selected_q])
    else:
        if st.button("🗑️ Clear chat history", key="clear_chat_render"):
            st.session_state["chat_history"] = []
            st.rerun()

        for msg in st.session_state["chat_history"]:
            with st.chat_message(
                msg["role"], avatar="🤖" if msg["role"] == "assistant" else None
            ):
                st.markdown(msg["content"])

        if prompt := st.chat_input(
            "Ask about Bengaluru traffic data…", key="chat_input_render"
        ):
            st.session_state["chat_history"].append(
                {"role": "user", "content": prompt}
            )
            with st.chat_message("user"):
                st.markdown(prompt)

            history_before = st.session_state["chat_history"][:-1]
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Groq is thinking…"):
                    response = _cached_call(
                        api_key, system_prompt, prompt, history_before
                    )
                st.markdown(response)

            st.session_state["chat_history"].append(
                {"role": "assistant", "content": response}
            )

    st.divider()

    # ── 2. Dispatch plan ───────────────────────────────────────────────────────
    with st.expander("📋 Generate station dispatch plan", expanded=False):
        _render_dispatch(df, api_key, system_prompt)

    # ── 3. Executive briefing ──────────────────────────────────────────────────
    with st.expander(
        "📄 Generate patrol briefing (for BTP Commissioner)", expanded=False
    ):
        _render_exec_briefing(df, api_key, system_prompt)

    # ── 4. RF Risk Forecaster ──────────────────────────────────────────────────
    with st.expander("📈 ML congestion risk forecaster", expanded=False):
        tab_tactical_ai.render(df)
