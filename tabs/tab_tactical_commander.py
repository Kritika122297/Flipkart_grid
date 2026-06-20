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
    top3 = df.groupby("police_station")["cis"].mean().nlargest(3)
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


def _call_gemini(api_key: str, system_prompt: str, user_msg: str, history: list) -> str:
    try:
        import google.generativeai as genai
    except ImportError:
        return (
            "❌ `google-generativeai` not installed.\n\n"
            "Run: `pip install google-generativeai` then reload the app."
        )
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system_prompt,
        )
        gemini_history = [
            {
                "role": "user" if m["role"] == "user" else "model",
                "parts": [m["content"]],
            }
            for m in history
        ]
        chat = model.start_chat(history=gemini_history)
        return chat.send_message(user_msg).text
    except Exception as exc:
        return f"❌ Gemini error: {exc}"


def _cached_call(api_key: str, system_prompt: str, user_msg: str, history: list) -> str:
    if "gemini_cache" not in st.session_state:
        st.session_state["gemini_cache"] = {}

    cache_key = user_msg.strip().lower()[:300]
    entry = st.session_state["gemini_cache"].get(cache_key)
    if entry and time.time() - entry[0] < 60:
        return entry[1]

    response = _call_gemini(api_key, system_prompt, user_msg, history)
    st.session_state["gemini_cache"][cache_key] = (time.time(), response)
    return response


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION RENDERS
# ══════════════════════════════════════════════════════════════════════════════

def render_chat_panel(df: pd.DataFrame) -> None:
    """Width-agnostic chat panel — safe to call from any tab context or app.py."""
    api_key: str = st.session_state.get("gemini_api_key", "")
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
            "Gemini API key not configured in the environment — showing demo responses."
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

    # ── Live Gemini chat ───────────────────────────────────────────────────────
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
            with st.spinner("Gemini is thinking…"):
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


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def render(df: pd.DataFrame) -> None:
    st.markdown(
        "<div style='text-align:center;padding:10px 0 30px;'>"
        "<h2 style='background:linear-gradient(90deg,#818CF8,#38BDF8,#A78BFA);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        "font-size:2.2rem;font-weight:800;margin-bottom:6px;'>"
        "🤖 Tactical AI Commander"
        "</h2>"
        "<p style='color:#888;font-size:1rem;margin:0;'>"
        "Gemini-powered intelligence · Dispatch planning · ML risk forecasting"
        "</p></div>",
        unsafe_allow_html=True,
    )

    api_key: str = st.session_state.get("gemini_api_key", "")
    system_prompt = _build_system_prompt(df)

    _render_dispatch(df, api_key, system_prompt)
    st.divider()
    _render_rf(df)
