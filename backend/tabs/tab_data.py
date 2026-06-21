"""
tabs/tab_data.py — Data upload, cleaning report, and full EDA module.

Sections
--------
1. Upload / Landing  — file uploader + feature highlights
2. Cleaning Report   — cards showing what was dropped and why
3. EDA (inner tabs)
   a. 📋 Summary      — shape, memory, dtypes, column info table
   b. 🔍 Data Quality — missing-value heatmap, null %, duplicates, dtype pie
   c. 📈 Statistics   — descriptive stats, correlation matrix, skew/kurtosis
   d. 📊 Visualize    — interactive histogram, boxplot, scatter
"""

from __future__ import annotations
import io
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from charts.utils import style_fig, ACCENT_SEQ, PLOTLY_TEMPLATE, CHART_BG

# Columns meaningful for numerical EDA (exclude raw IDs and list columns)
_EDA_NUM_COLS = [
    "hour", "num_violations", "violation_severity",
    "vehicle_size_score", "time_factor", "cis",
    "latitude", "longitude", "is_rush_hour", "is_weekend",
    "near_junction", "junction_factor",
]

def _hashable(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with list-valued columns dropped (they are unhashable)."""
    list_cols = [c for c in df.columns
                 if df[c].apply(lambda x: isinstance(x, list)).any()]
    return df.drop(columns=list_cols, errors="ignore")


# ── Cleaning constants ────────────────────────────────────────────────
_PROTECTED_COLS = {
    "cis", "hour", "day_of_week", "day_num", "month", "month_name",
    "is_rush_hour", "is_weekend", "latitude", "longitude",
    "created_datetime", "police_station", "violation_list",
    "num_violations", "violation_severity", "vehicle_size_score",
    "time_factor", "junction_factor", "near_junction",
}
_TEXT_COLS = ["police_station", "location", "vehicle_type", "violation_type", "junction_name"]


@st.cache_data(show_spinner=False)
def _baseline_stats(_df: pd.DataFrame, _key: str = "") -> dict:
    """Compute expensive stats once per upload (cached by upload key)."""
    h = _hashable(_df)
    return {
        "rows":      len(_df),
        "cols":      _df.shape[1],
        "nulls":     int(_df.isnull().sum().sum()),
        "dupes":     int(h.duplicated().sum()),
        "null_pct":  _df.isnull().mean().to_dict(),
        "num_nulls": int(_df.select_dtypes(include="number").isnull().sum().sum()),
        "cat_nulls": int(_df.select_dtypes(include="object").isnull().sum().sum()),
    }


def _apply_cleaning(df: pd.DataFrame, opts: dict) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    log = []

    if opts["drop_null_cols"]:
        thr   = opts["null_threshold"] / 100
        drop  = [c for c, p in df.isnull().mean().items()
                 if p > thr and c not in _PROTECTED_COLS]
        if drop:
            out = out.drop(columns=drop)
            log.append(f"Dropped **{len(drop)}** column(s) with >{opts['null_threshold']}% nulls: "
                       + ", ".join(f"`{c}`" for c in drop))

    if opts["fill_numeric"]:
        total = 0
        for col in out.select_dtypes(include="number").columns:
            n = int(out[col].isnull().sum())
            if n:
                v = out[col].median() if opts["numeric_method"] == "Median" else out[col].mean()
                out[col] = out[col].fillna(v)
                total += n
        if total:
            log.append(f"Filled **{total:,}** missing numeric values with {opts['numeric_method'].lower()}")

    if opts["fill_categorical"]:
        total = 0
        for col in out.select_dtypes(include="object").columns:
            n = int(out[col].isnull().sum())
            if n:
                modes = out[col].mode()
                v = (modes.iloc[0] if opts["categorical_method"] == "Mode" and len(modes)
                     else "Unknown")
                out[col] = out[col].fillna(v)
                total += n
        if total:
            log.append(f"Filled **{total:,}** missing text values with {opts['categorical_method'].lower()}")

    if opts["remove_dupes"]:
        list_cols = [c for c in out.columns if out[c].apply(lambda x: isinstance(x, list)).any()]
        sub       = [c for c in out.columns if c not in list_cols]
        before    = len(out)
        out       = out.drop_duplicates(subset=sub)
        n         = before - len(out)
        if n:
            log.append(f"Removed **{n:,}** duplicate rows")

    if opts["standardize_text"]:
        cols = [c for c in _TEXT_COLS if c in out.columns]
        for c in cols:
            out[c] = out[c].astype(str).str.strip().str.upper()
        log.append(f"Standardized text in **{len(cols)}** column(s) (trim + UPPERCASE)")

    return out, log


def _render_cleaning_panel(df: pd.DataFrame) -> None:
    st.markdown(
        "<p class='section-header'>🧹 Interactive Data Cleaning</p>"
        "<p class='section-sub'>Configure rules, preview the impact, then apply in one click</p>",
        unsafe_allow_html=True,
    )

    upload_key = st.session_state.get("_processed_upload", "")
    base       = _baseline_stats(df, upload_key)
    is_applied = st.session_state.get("_cleaning_applied", False)

    # ── Applied state ─────────────────────────────────────────────────
    if is_applied:
        changes    = st.session_state.get("_cleaning_changes", [])
        clean_base = _baseline_stats(st.session_state.df,
                                     upload_key + "_cleaned")
        st.markdown(
            f"""<div style="background:linear-gradient(135deg,rgba(0,210,100,0.12),
            rgba(0,210,255,0.08)); border:1px solid rgba(0,210,100,0.3);
            border-radius:14px; padding:18px 24px; margin-bottom:16px;">
              <div style="color:#00D264; font-weight:700; font-size:1rem; margin-bottom:10px;">
                ✅ Cleaning applied — {len(changes)} step(s) completed
              </div>
              {"".join(f"<div style='color:#aaa;font-size:0.87rem;margin-top:5px;'>• {c}</div>" for c in changes)}
            </div>""",
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows",    f"{clean_base['rows']:,}",
                  delta=f"{clean_base['rows'] - base['rows']:,}", delta_color="inverse")
        c2.metric("Columns", f"{clean_base['cols']:,}",
                  delta=f"{clean_base['cols'] - base['cols']:,}", delta_color="inverse")
        c3.metric("Nulls",   f"{clean_base['nulls']:,}",
                  delta=f"{clean_base['nulls'] - base['nulls']:,}", delta_color="inverse")
        c4.metric("Dupes",   f"{clean_base['dupes']:,}",
                  delta=f"{clean_base['dupes'] - base['dupes']:,}", delta_color="inverse")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset to Original Data", key="_cl_reset"):
            st.session_state.df = st.session_state.get("raw_df", df)
            st.session_state._cleaning_applied = False
            st.session_state._cleaning_changes = []
        return

    # ── Controls + live preview ───────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("#### ⚙️ Cleaning Steps")

        drop_null   = st.checkbox("Drop high-null columns",            value=True,  key="_cl_drop")
        null_thr    = st.slider("Null % threshold", 10, 90, 50, 5,    key="_cl_thr",
                                help="Drop columns where null% exceeds this value",
                                disabled=not drop_null)

        fill_num    = st.checkbox("Fill missing numeric values",        value=True,  key="_cl_num")
        num_method  = st.radio("Numeric fill", ["Median", "Mean"],
                               horizontal=True, key="_cl_nmeth", disabled=not fill_num)

        fill_cat    = st.checkbox("Fill missing text/category values",  value=True,  key="_cl_cat")
        cat_method  = st.radio("Text fill", ["Mode", "Unknown"],
                               horizontal=True, key="_cl_cmeth", disabled=not fill_cat)

        rm_dupes    = st.checkbox("Remove duplicate rows",              value=True,  key="_cl_dup")
        std_text    = st.checkbox("Standardize text (trim + UPPERCASE)",value=False, key="_cl_std")

    opts = {
        "drop_null_cols":     drop_null,
        "null_threshold":     null_thr,
        "fill_numeric":       fill_num,
        "numeric_method":     num_method,
        "fill_categorical":   fill_cat,
        "categorical_method": cat_method,
        "remove_dupes":       rm_dupes,
        "standardize_text":   std_text,
    }

    # ── Fast preview (uses cached baseline, no full df copy) ──────────
    with right:
        st.markdown("#### 📊 Impact Preview")

        est_rows  = base["rows"]
        est_cols  = base["cols"]
        est_nulls = base["nulls"]
        est_dupes = base["dupes"]
        will_do   = []

        if drop_null:
            thr       = null_thr / 100
            drop_list = [c for c, p in base["null_pct"].items()
                         if p > thr and c not in _PROTECTED_COLS]
            if drop_list:
                col_nulls  = int(sum(df[c].isnull().sum() for c in drop_list if c in df.columns))
                est_cols  -= len(drop_list)
                est_nulls -= col_nulls
                will_do.append(f"Drop **{len(drop_list)}** column(s) with >{null_thr}% nulls")

        if fill_num and base["num_nulls"]:
            est_nulls -= base["num_nulls"]
            will_do.append(f"Fill **{base['num_nulls']:,}** missing numeric values with {num_method.lower()}")

        if fill_cat and base["cat_nulls"]:
            est_nulls -= base["cat_nulls"]
            will_do.append(f"Fill **{base['cat_nulls']:,}** missing text values with {cat_method.lower()}")

        if rm_dupes and base["dupes"]:
            est_rows  -= base["dupes"]
            est_dupes  = 0
            will_do.append(f"Remove **{base['dupes']:,}** duplicate rows")

        if std_text:
            n = len([c for c in _TEXT_COLS if c in df.columns])
            will_do.append(f"Standardize text in **{n}** column(s)")

        def _d(a, b): return f"{a-b:,}" if a != b else None
        # Before / After as stacked markdown to avoid nested columns (Streamlit 1.45 restriction)
        st.markdown(
            "<div style='background:rgba(255,255,255,0.03);border-radius:10px;"
            "padding:12px 14px;margin-bottom:8px;'>"
            "<div style='color:#888;font-size:0.73rem;font-weight:700;"
            "letter-spacing:1px;margin-bottom:8px;'>BEFORE</div>"
            f"<div style='color:#ccc;font-size:0.88rem;line-height:2;'>"
            f"Rows: <b>{base['rows']:,}</b> &nbsp;·&nbsp; "
            f"Columns: <b>{base['cols']:,}</b> &nbsp;·&nbsp; "
            f"Nulls: <b>{base['nulls']:,}</b> &nbsp;·&nbsp; "
            f"Dupes: <b>{base['dupes']:,}</b></div></div>"
            "<div style='background:rgba(0,210,100,0.06);border-radius:10px;"
            "padding:12px 14px;'>"
            "<div style='color:#00D264;font-size:0.73rem;font-weight:700;"
            "letter-spacing:1px;margin-bottom:8px;'>AFTER (estimated)</div>"
            f"<div style='color:#ccc;font-size:0.88rem;line-height:2;'>"
            f"Rows: <b>{est_rows:,}</b> &nbsp;·&nbsp; "
            f"Columns: <b>{est_cols:,}</b> &nbsp;·&nbsp; "
            f"Nulls: <b>{max(0,est_nulls):,}</b> &nbsp;·&nbsp; "
            f"Dupes: <b>{est_dupes:,}</b></div></div>",
            unsafe_allow_html=True,
        )

        if will_do:
            st.markdown(
                "<div style='margin-top:10px;'>" +
                "".join(f"<div style='color:#aaa;font-size:0.83rem;margin-top:5px;'>"
                        f"✓ {s}</div>" for s in will_do) +
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No changes with current settings.")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.button("🧹 Apply Cleaning", type="primary", key="_cl_apply"):
        cleaned, changes = _apply_cleaning(
            st.session_state.get("raw_df", df), opts
        )
        st.session_state.df = cleaned
        st.session_state._cleaning_applied = True
        st.session_state._cleaning_changes = changes


# Columns shown in the raw data preview
_PREVIEW_COLS = [
    "created_datetime", "police_station", "location",
    "vehicle_type", "violation_type", "latitude", "longitude",
    "violation_severity", "cis",
]


# ════════════════════════════════════════════════════════════════════
# EXPORT / DOWNLOAD
# ════════════════════════════════════════════════════════════════════

def _render_export(df: pd.DataFrame) -> None:
    with st.expander("📥 Export & Download", expanded=False):
        st.markdown(
            "<div class='pw-info-banner' style='margin-top:4px;'>ℹ️ Download the current "
            "(possibly cleaned) dataset or a full Excel report with multiple analysis sheets."
            "</div>",
            unsafe_allow_html=True,
        )

        col_csv, col_xl = st.columns(2)

        with col_csv:
            st.markdown("**CSV — Current Dataset**")
            st.caption("Downloads the active dataset (after any cleaning steps applied).")
            csv_bytes = df.drop(
                columns=[c for c in df.columns
                         if df[c].apply(lambda x: isinstance(x, list)).any()],
                errors="ignore",
            ).to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV",
                data=csv_bytes,
                file_name="parkwatch_data.csv",
                mime="text/csv",
                use_container_width=True,
                key="exp_csv",
            )

        with col_xl:
            st.markdown("**Excel — Full Analysis Report**")
            st.caption("5-sheet workbook: summary stats, top hotspots, "
                       "enforcement rankings, hourly pattern, raw data (5k rows).")
            try:
                xl_buf = _build_excel_report(df)  # cached — fast on repeat renders
                st.download_button(
                    "⬇️ Download Excel Report",
                    data=xl_buf,
                    file_name="parkwatch_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="exp_xl_dl",
                )
            except Exception as e:
                st.markdown(
                    f"<div class='pw-warn-banner'>⚠️ Excel export requires openpyxl: "
                    f"<code>pip install openpyxl</code><br>"
                    f"<small style='color:#666;'>{e}</small></div>",
                    unsafe_allow_html=True,
                )


@st.cache_data(show_spinner=False)
def _build_excel_report(_df: pd.DataFrame) -> bytes:
    df = _df  # alias — _df skips hashing, df used internally
    safe_df = df.drop(
        columns=[c for c in df.columns
                 if df[c].apply(lambda x: isinstance(x, list)).any()],
        errors="ignore",
    )

    # Sheet 1: Summary stats
    summary = pd.DataFrame({
        "Metric": [
            "Total Violations", "Unique Stations", "Unique Locations",
            "Avg CIS Score", "Median CIS Score", "High-Risk Violations (CIS>50)",
            "Rush Hour Violations", "Weekend Violations", "Near-Junction Violations",
        ],
        "Value": [
            len(df),
            df["police_station"].nunique() if "police_station" in df.columns else "N/A",
            df["location"].nunique()       if "location"        in df.columns else "N/A",
            round(float(df["cis"].mean()), 2)   if "cis" in df.columns else "N/A",
            round(float(df["cis"].median()), 2) if "cis" in df.columns else "N/A",
            int((df["cis"] > 50).sum())         if "cis" in df.columns else "N/A",
            int(df["is_rush_hour"].sum())  if "is_rush_hour" in df.columns else "N/A",
            int(df["is_weekend"].sum())    if "is_weekend"   in df.columns else "N/A",
            int(df["near_junction"].sum()) if "near_junction" in df.columns else "N/A",
        ],
    })

    # Sheet 2: Top hotspots
    hotspots = pd.DataFrame()
    if "location" in df.columns and "cis" in df.columns:
        hotspots = (
            df.groupby("location", observed=True)
            .agg(violations=("cis", "size"), avg_cis=("cis", "mean"))
            .reset_index()
            .sort_values("violations", ascending=False)
            .head(20)
        )
        hotspots["avg_cis"] = hotspots["avg_cis"].round(1)

    # Sheet 3: Enforcement ranking
    enforcement = pd.DataFrame()
    if "police_station" in df.columns and "cis" in df.columns:
        enforcement = (
            df.groupby("police_station", observed=True)
            .agg(violations=("cis", "size"), avg_cis=("cis", "mean"),
                 total_cis=("cis", "sum"))
            .reset_index()
            .sort_values("total_cis", ascending=False)
            .head(30)
        )
        enforcement["avg_cis"]   = enforcement["avg_cis"].round(1)
        enforcement["total_cis"] = enforcement["total_cis"].round(1)

    # Sheet 4: Hourly pattern
    hourly = pd.DataFrame()
    if "hour" in df.columns and "cis" in df.columns:
        hourly = (
            df.groupby("hour")
            .agg(violations=("cis", "size"), avg_cis=("cis", "mean"))
            .reset_index()
        )
        hourly["avg_cis"] = hourly["avg_cis"].round(1)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        summary.to_excel(writer,    sheet_name="Summary",    index=False)
        if not hotspots.empty:
            hotspots.to_excel(writer,  sheet_name="Top Hotspots", index=False)
        if not enforcement.empty:
            enforcement.to_excel(writer, sheet_name="Enforcement",  index=False)
        if not hourly.empty:
            hourly.to_excel(writer,    sheet_name="Hourly Pattern", index=False)
        safe_df.head(5000).to_excel(writer, sheet_name="Raw Data (5000 rows)", index=False)

    return buf.getvalue()


# ════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ════════════════════════════════════════════════════════════════════

def render(df: pd.DataFrame | None, stats: dict | None) -> None:
    if df is None:
        st.markdown(
            "<div class='pw-empty-state'>"
            "<div class='es-icon'>📂</div>"
            "<div class='es-title'>No data loaded yet</div>"
            "<div class='es-sub'>Use the <b style='color:#6C63FF;'>sidebar uploader</b> "
            "to drop your BTP violation CSV — the full EDA panel unlocks instantly.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    _success_banner(stats)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    _cleaning_report(stats)
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # Use raw_df as base for the cleaning panel so Reset always works
    raw_df = st.session_state.get("raw_df", df)
    _render_cleaning_panel(raw_df)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    # EDA uses the (possibly cleaned) df from session state
    active_df = st.session_state.get("df", df)
    _render_export(active_df)
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    _eda_section(active_df)


# ════════════════════════════════════════════════════════════════════
# UPLOAD HELPERS
# ════════════════════════════════════════════════════════════════════

def _show_upload_landing() -> None:
    st.markdown(
        "<p class='section-header'>📂 Load Your Dataset</p>"
        "<p class='section-sub'>Upload the BTP parking violation CSV to unlock the full dashboard</p>",
        unsafe_allow_html=True,
    )
    features = [
        ("🧹", "Auto Cleaning",
         "Bad timestamps & out-of-bounds coordinates removed automatically"),
        ("📊", "Full EDA",
         "Missing values, distributions, correlations, and statistics generated instantly"),
        ("🚀", "Full Dashboard",
         "All 5 analysis tabs unlock once data is loaded"),
    ]
    for col, (icon, title, desc) in zip(st.columns(3), features):
        col.markdown(
            f"""<div class='glass-panel' style='text-align:center; padding:28px 16px;'>
                <div style='font-size:2rem; margin-bottom:10px;'>{icon}</div>
                <div style='color:#E8E8E8; font-weight:700; margin-bottom:6px;'>{title}</div>
                <div style='color:#666; font-size:0.83rem;'>{desc}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    _upload_widget()


def _on_file_uploaded() -> None:
    """on_change callback — sets ONLY a flag, zero disk I/O.
    Callbacks run in the asyncio event loop thread; any blocking I/O here
    prevents Tornado from sending WebSocket heartbeats and drops the connection.
    All file writing and CSV processing is deferred to app.py's script thread."""
    uploaded = st.session_state.get("_file_uploader_widget")
    if uploaded is None:
        return
    upload_key = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("_processed_upload") == upload_key:
        return
    st.session_state["_pending_upload_key"] = upload_key


def _upload_widget() -> None:
    st.file_uploader(
        "Drop your BTP violation CSV here",
        type=["csv"],
        key="_file_uploader_widget",
        on_change=_on_file_uploaded,
        help="Expected columns: id, police_station, location, vehicle_type, "
             "violation_type, junction_name, latitude, longitude, created_datetime",
    )
    err = st.session_state.pop("_upload_error", None)
    if err:
        st.error(f"Error processing file: {err}")


# ════════════════════════════════════════════════════════════════════
# SUCCESS BANNER
# ════════════════════════════════════════════════════════════════════

def _success_banner(stats: dict) -> None:
    st.markdown(
        f"""<div style='background:linear-gradient(135deg,rgba(0,210,100,0.12),rgba(0,210,255,0.08));
                        border:1px solid rgba(0,210,100,0.3); border-radius:14px;
                        padding:16px 24px; margin-bottom:20px;
                        display:flex; align-items:center; gap:14px;'>
            <span style='font-size:1.8rem;'>✅</span>
            <div>
                <div style='color:#00D264; font-weight:700; font-size:1rem;'>
                    Dataset loaded successfully
                </div>
                <div style='color:#666; font-size:0.82rem;'>
                    {stats["final_rows"]:,} clean records &nbsp;·&nbsp;
                    {stats["date_min"]} → {stats["date_max"]}
                </div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
# CLEANING REPORT
# ════════════════════════════════════════════════════════════════════

def _cleaning_report(stats: dict) -> None:
    st.markdown(
        "<p class='section-header'>🧹 Data Cleaning Report</p>"
        "<p class='section-sub'>Steps applied automatically before any analysis</p>",
        unsafe_allow_html=True,
    )
    retention = stats["final_rows"] / stats["raw_rows"] * 100
    c1, c2, c3, c4 = st.columns(4)
    _kpi_card(c1, "Raw Rows Ingested",   f'{stats["raw_rows"]:,}',              "From uploaded CSV",           "kpi-card")
    _kpi_card(c2, "Clean Rows Kept",     f'{stats["final_rows"]:,}',            f'{retention:.1f}% retention', "savings-card")
    _kpi_card(c3, "Bad Datetime Dropped",f'{stats["dropped_bad_datetime"]:,}',  "Unparseable timestamps",      "impact-card")
    _kpi_card(c4, "Out-of-Bounds Coords",f'{stats["dropped_invalid_coords"]:,}',"Outside Bengaluru metro",     "impact-card")


def _kpi_card(col, label: str, value: str, sub: str, css_class: str) -> None:
    col.markdown(
        f"""<div class='{css_class}'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{value}</div>
            <div class='kpi-sub'>{sub}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
# EDA SECTION  (inner tabs)
# ════════════════════════════════════════════════════════════════════

def _eda_section(df: pd.DataFrame) -> None:
    st.markdown(
        "<p class='section-header'>📊 Exploratory Data Analysis</p>"
        "<p class='section-sub'>Auto-generated deep-dive into your dataset</p>",
        unsafe_allow_html=True,
    )
    t1, t2, t3, t4 = st.tabs([
        "📋 Summary",
        "🔍 Data Quality",
        "📈 Statistics",
        "📊 Visualize",
    ])
    with t1:
        _render_summary(df)
    with t2:
        _render_quality(df)
    with t3:
        _render_statistics(df)
    with t4:
        _render_visualizations(df)


# ────────────────────────────────────────────────────────────────────
# TAB 1 — SUMMARY
# ────────────────────────────────────────────────────────────────────

def _render_summary(df: pd.DataFrame) -> None:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Top metric cards ──
    total_missing = int(df.isnull().sum().sum())
    missing_pct   = total_missing / (df.shape[0] * df.shape[1]) * 100
    duplicates    = int(_hashable(df).duplicated().sum())
    mem_mb        = df.memory_usage(deep=True).sum() / 1024 ** 2

    cols = st.columns(5)
    _metric_card(cols[0], "Rows",           f"{df.shape[0]:,}",       "#6C63FF")
    _metric_card(cols[1], "Columns",        f"{df.shape[1]}",          "#00D2FF")
    _metric_card(cols[2], "Missing Cells",  f"{total_missing:,}",      "#FF4B4B")
    _metric_card(cols[3], "Duplicate Rows", f"{duplicates:,}",         "#FFD93D")
    _metric_card(cols[4], "Memory Usage",   f"{mem_mb:.1f} MB",        "#6BCB77")

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Missing % banner ──
    color = "#FF4B4B" if missing_pct > 10 else "#FFD93D" if missing_pct > 2 else "#00D264"
    st.markdown(
        f"""<div style='background:rgba(30,33,48,0.55); border:1px solid rgba(108,99,255,0.12);
                        border-radius:12px; padding:14px 20px; margin-bottom:20px;
                        display:flex; align-items:center; gap:10px;'>
            <span style='color:{color}; font-size:1.2rem; font-weight:800;'>{missing_pct:.2f}%</span>
            <span style='color:#888; font-size:0.87rem;'>
                of all cells contain missing values
                ({total_missing:,} / {df.shape[0] * df.shape[1]:,} cells)
            </span>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Column info table ──
    st.markdown(
        "<p style='color:#888; font-weight:600; font-size:0.9rem; margin-bottom:8px;'>"
        "Column Overview</p>",
        unsafe_allow_html=True,
    )
    col_info = _build_column_info(df)
    st.dataframe(col_info, use_container_width=True, height=420)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Data preview ──
    st.markdown(
        "<p style='color:#888; font-weight:600; font-size:0.9rem; margin-bottom:8px;'>"
        "Raw Data Preview (first 100 rows)</p>",
        unsafe_allow_html=True,
    )
    show = [c for c in _PREVIEW_COLS if c in df.columns]
    st.dataframe(df[show].head(100), use_container_width=True, height=320)


def _metric_card(col, label: str, value: str, color: str) -> None:
    col.markdown(
        f"""<div style='background:rgba(30,33,48,0.6);
                        border:1px solid {color}33;
                        border-top:3px solid {color};
                        border-radius:12px; padding:18px 14px; text-align:center;'>
            <div style='font-size:0.75rem; font-weight:600; color:#888;
                        text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;'>
                {label}
            </div>
            <div style='font-size:1.55rem; font-weight:800; color:{color};'>
                {value}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


@st.cache_data
def _build_column_info(_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in _df.columns:
        series = _df[col]
        non_null = int(series.notna().sum())
        null_cnt = int(series.isna().sum())
        null_pct = round(null_cnt / len(_df) * 100, 2)
        try:
            unique = int(series.nunique())
        except TypeError:
            unique = None  # list-valued column — Arrow needs uniform type
        rows.append({
            "Column": col,
            "Dtype": str(series.dtype),
            "Non-Null": non_null,
            "Null Count": null_cnt,
            "Null %": null_pct,
            "Unique Values": unique,
        })
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────
# TAB 2 — DATA QUALITY
# ────────────────────────────────────────────────────────────────────

def _render_quality(df: pd.DataFrame) -> None:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Null % bar chart + Missing pattern heatmap ──
    q1, q2 = st.columns(2)
    with q1:
        _null_bar_chart(df)
    with q2:
        _missing_heatmap(df)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Duplicate rows ──
    dup_count = int(_hashable(df).duplicated().sum())
    dup_color = "#FF4B4B" if dup_count > 0 else "#00D264"
    dup_icon  = "⚠️" if dup_count > 0 else "✅"
    st.markdown(
        f"""<div style='background:rgba(30,33,48,0.55); border:1px solid rgba(108,99,255,0.12);
                        border-radius:12px; padding:16px 22px; margin-bottom:20px;'>
            <span style='font-size:1.1rem;'>{dup_icon}</span>
            <span style='color:{dup_color}; font-weight:700; margin-left:8px;'>
                {dup_count:,} duplicate rows detected
            </span>
            <span style='color:#666; font-size:0.85rem; margin-left:10px;'>
                ({dup_count / len(df) * 100:.3f}% of dataset)
            </span>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Dtype distribution ──
    d1, d2 = st.columns(2)
    with d1:
        _dtype_pie(df)
    with d2:
        _completeness_chart(df)


def _null_bar_chart(df: pd.DataFrame) -> None:
    null_series = df.isnull().mean() * 100
    null_series = null_series[null_series > 0].sort_values(ascending=True)

    if null_series.empty:
        st.markdown(
            "<div class='glass-panel' style='text-align:center; padding:50px;'>"
            "<div style='font-size:2rem;'>✅</div>"
            "<div style='color:#00D264; font-weight:600; margin-top:10px;'>"
            "No missing values across all columns</div></div>",
            unsafe_allow_html=True,
        )
        return

    colors = [
        "#FF4B4B" if v > 20 else "#FFD93D" if v > 5 else "#6C63FF"
        for v in null_series.values
    ]
    fig = go.Figure(go.Bar(
        x=null_series.values,
        y=null_series.index,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in null_series.values],
        textposition="outside",
        textfont=dict(size=10, color="#aaa"),
    ))
    fig.update_layout(
        title=dict(text="Null % by Column", font=dict(size=14, color="#ddd")),
        xaxis=dict(title="Missing %", range=[0, null_series.max() * 1.35]),
        yaxis_title=None,
    )
    style_fig(fig, 360)
    st.plotly_chart(fig, use_container_width=True)


def _missing_heatmap(df: pd.DataFrame) -> None:
    # Sample 300 rows for a readable pattern heatmap
    sample = df.sample(min(300, len(df)), random_state=42)
    raw_cols = [c for c in _PREVIEW_COLS if c in sample.columns]
    matrix = sample[raw_cols].isnull().astype(int)

    fig = go.Figure(go.Heatmap(
        z=matrix.values.T,
        x=[f"Row {i}" for i in range(len(matrix))],
        y=raw_cols,
        colorscale=[[0, "rgba(108,99,255,0.08)"], [1, "#FF4B4B"]],
        showscale=False,
        hovertemplate="Column: %{y}<br>Missing: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Missing Pattern (300-row sample)", font=dict(size=14, color="#ddd")),
        xaxis=dict(showticklabels=False, title="Sampled Rows"),
        yaxis_title=None,
    )
    style_fig(fig, 360)
    st.plotly_chart(fig, use_container_width=True)


def _dtype_pie(df: pd.DataFrame) -> None:
    dtype_counts: dict[str, int] = {}
    for dtype in df.dtypes:
        key = str(dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + 1

    fig = go.Figure(go.Pie(
        labels=list(dtype_counts.keys()),
        values=list(dtype_counts.values()),
        hole=0.5,
        marker=dict(colors=ACCENT_SEQ[: len(dtype_counts)]),
        textinfo="label+value",
        textfont=dict(size=11, color="#ddd"),
    ))
    fig.update_layout(
        title=dict(text="Column Data Types", font=dict(size=14, color="#ddd")),
        showlegend=True,
    )
    style_fig(fig, 340)
    st.plotly_chart(fig, use_container_width=True)


def _completeness_chart(df: pd.DataFrame) -> None:
    completeness = ((1 - df.isnull().mean()) * 100).sort_values()
    colors = [
        "#FF4B4B" if v < 80 else "#FFD93D" if v < 95 else "#00D264"
        for v in completeness.values
    ]
    fig = go.Figure(go.Bar(
        x=completeness.values,
        y=completeness.index,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in completeness.values],
        textposition="inside",
        textfont=dict(size=9, color="#fff"),
    ))
    fig.update_layout(
        title=dict(text="Column Completeness %", font=dict(size=14, color="#ddd")),
        xaxis=dict(title="Completeness %", range=[0, 110]),
        yaxis_title=None,
    )
    style_fig(fig, 340)
    st.plotly_chart(fig, use_container_width=True)


# ────────────────────────────────────────────────────────────────────
# TAB 3 — STATISTICS
# ────────────────────────────────────────────────────────────────────

def _render_statistics(df: pd.DataFrame) -> None:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    num_df = _get_num_df(df)

    # ── Descriptive stats ──
    st.markdown(
        "<p style='color:#888; font-weight:600; font-size:0.9rem; margin-bottom:8px;'>"
        "Descriptive Statistics — Numerical Features</p>",
        unsafe_allow_html=True,
    )
    desc = num_df.describe().T.round(3)
    desc.index.name = "Feature"
    st.dataframe(desc, use_container_width=True, height=360)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Skewness & Kurtosis ──
    s1, s2 = st.columns(2)
    with s1:
        _skew_chart(num_df)
    with s2:
        _kurtosis_chart(num_df)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Correlation matrix ──
    st.markdown(
        "<p style='color:#888; font-weight:600; font-size:0.9rem; margin-bottom:8px;'>"
        "Correlation Matrix</p>",
        unsafe_allow_html=True,
    )
    _correlation_heatmap(num_df)


@st.cache_data
def _get_num_df(_df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in _EDA_NUM_COLS if c in _df.columns]
    return _df[available].select_dtypes(include="number")


def _skew_chart(num_df: pd.DataFrame) -> None:
    skew = num_df.skew().sort_values()
    colors = [
        "#FF4B4B" if abs(v) > 1 else "#FFD93D" if abs(v) > 0.5 else "#6C63FF"
        for v in skew.values
    ]
    fig = go.Figure(go.Bar(
        x=skew.values,
        y=skew.index,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.2f}" for v in skew.values],
        textposition="outside",
        textfont=dict(size=9, color="#aaa"),
    ))
    fig.update_layout(
        title=dict(text="Skewness (|>1| = highly skewed)", font=dict(size=13, color="#ddd")),
        xaxis_title="Skewness",
        yaxis_title=None,
    )
    style_fig(fig, 360)
    st.plotly_chart(fig, use_container_width=True)


def _kurtosis_chart(num_df: pd.DataFrame) -> None:
    kurt = num_df.kurtosis().sort_values()
    colors = [
        "#FF4B4B" if abs(v) > 3 else "#FFD93D" if abs(v) > 1 else "#6C63FF"
        for v in kurt.values
    ]
    fig = go.Figure(go.Bar(
        x=kurt.values,
        y=kurt.index,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.2f}" for v in kurt.values],
        textposition="outside",
        textfont=dict(size=9, color="#aaa"),
    ))
    fig.update_layout(
        title=dict(text="Kurtosis (|>3| = heavy tails)", font=dict(size=13, color="#ddd")),
        xaxis_title="Kurtosis",
        yaxis_title=None,
    )
    style_fig(fig, 360)
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def _compute_corr(_num_df: pd.DataFrame) -> pd.DataFrame:
    return _num_df.corr().round(2)


def _correlation_heatmap(num_df: pd.DataFrame) -> None:
    corr = _compute_corr(num_df)
    cols = list(corr.columns)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=cols,
        y=cols,
        colorscale=[
            [0.0,  "#FF4B4B"],
            [0.25, "#FF9944"],
            [0.5,  "#1a1a2e"],
            [0.75, "#6C63FF"],
            [1.0,  "#00D2FF"],
        ],
        zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=9, color="#ddd"),
        hovertemplate="%{y} × %{x}: <b>%{z}</b><extra></extra>",
        colorbar=dict(title="r", tickfont=dict(color="#888")),
    ))
    fig.update_layout(
        title=dict(text="Pearson Correlation Matrix", font=dict(size=14, color="#ddd")),
        xaxis=dict(tickangle=-40),
        yaxis_title=None,
    )
    style_fig(fig, 480)
    st.plotly_chart(fig, use_container_width=True)


# ────────────────────────────────────────────────────────────────────
# TAB 4 — VISUALIZATIONS
# ────────────────────────────────────────────────────────────────────

def _render_visualizations(df: pd.DataFrame) -> None:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    num_df = _get_num_df(df)
    num_cols = list(num_df.columns)

    # ── Histogram ──
    st.markdown(
        "<p style='color:#888; font-weight:600; font-size:0.9rem;'>Distribution — Histogram</p>",
        unsafe_allow_html=True,
    )
    h1, h2 = st.columns([1, 3])
    with h1:
        hist_col = st.selectbox("Feature", num_cols, index=num_cols.index("cis") if "cis" in num_cols else 0, key="hist_col")
        n_bins   = st.slider("Bins", 10, 100, 40, key="hist_bins")
    with h2:
        _histogram(num_df, hist_col, n_bins)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Boxplot ──
    st.markdown(
        "<p style='color:#888; font-weight:600; font-size:0.9rem;'>Spread & Outliers — Box Plot</p>",
        unsafe_allow_html=True,
    )
    b1, b2 = st.columns([1, 3])
    with b1:
        box_col = st.selectbox("Feature", num_cols, index=num_cols.index("violation_severity") if "violation_severity" in num_cols else 0, key="box_col")
    with b2:
        _boxplot(num_df, box_col)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Scatter ──
    st.markdown(
        "<p style='color:#888; font-weight:600; font-size:0.9rem;'>Relationship — Scatter Plot</p>",
        unsafe_allow_html=True,
    )
    sc1, sc2, sc3 = st.columns([1, 1, 3])
    default_x = num_cols.index("cis") if "cis" in num_cols else 0
    default_y = num_cols.index("violation_severity") if "violation_severity" in num_cols else 1
    with sc1:
        x_col = st.selectbox("X axis", num_cols, index=default_x, key="sc_x")
    with sc2:
        y_col = st.selectbox("Y axis", num_cols, index=default_y, key="sc_y")
    with sc3:
        _scatter(num_df, x_col, y_col)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Multi-feature boxplot ──
    st.markdown(
        "<p style='color:#888; font-weight:600; font-size:0.9rem;'>All Features — Side-by-Side Box Plots</p>",
        unsafe_allow_html=True,
    )
    _multi_boxplot(num_df)


def _histogram(num_df: pd.DataFrame, col: str, bins: int) -> None:
    data = num_df[col].dropna()
    counts, edges = np.histogram(data, bins=bins)
    mids = (edges[:-1] + edges[1:]) / 2

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=mids,
        y=counts,
        marker=dict(
            color=counts,
            colorscale=[[0, "#3a2a6e"], [0.5, "#6C63FF"], [1, "#00D2FF"]],
            line=dict(width=0),
        ),
        hovertemplate="Value: %{x:.2f}<br>Count: %{y:,}<extra></extra>",
    ))
    # mean / median lines
    fig.add_vline(x=float(data.mean()),   line=dict(color="#FF4B4B", dash="dash", width=1.5),
                  annotation_text="mean",   annotation_font=dict(color="#FF4B4B", size=10))
    fig.add_vline(x=float(data.median()), line=dict(color="#FFD93D", dash="dot",  width=1.5),
                  annotation_text="median", annotation_font=dict(color="#FFD93D", size=10))
    fig.update_layout(
        title=dict(text=f"Distribution of {col}", font=dict(size=14, color="#ddd")),
        xaxis_title=col,
        yaxis_title="Frequency",
        bargap=0.02,
    )
    style_fig(fig, 380)
    st.plotly_chart(fig, use_container_width=True)


def _boxplot(num_df: pd.DataFrame, col: str) -> None:
    data = num_df[col].dropna()
    fig = go.Figure(go.Box(
        y=data,
        name=col,
        marker=dict(color="#6C63FF", size=3, opacity=0.4),
        line=dict(color="#00D2FF"),
        fillcolor="rgba(108,99,255,0.15)",
        boxmean="sd",
        hoverinfo="y",
    ))
    fig.update_layout(
        title=dict(text=f"Box Plot — {col}", font=dict(size=14, color="#ddd")),
        yaxis_title=col,
        showlegend=False,
    )
    style_fig(fig, 380)
    st.plotly_chart(fig, use_container_width=True)


def _scatter(num_df: pd.DataFrame, x_col: str, y_col: str) -> None:
    # Sample 5k points for speed
    sample = num_df[[x_col, y_col]].dropna().sample(min(5000, len(num_df)), random_state=42)
    corr_val = sample[x_col].corr(sample[y_col])

    fig = go.Figure(go.Scatter(
        x=sample[x_col],
        y=sample[y_col],
        mode="markers",
        marker=dict(
            color=sample[x_col],
            colorscale=[[0, "#3a2a6e"], [0.5, "#6C63FF"], [1, "#00D2FF"]],
            size=4,
            opacity=0.55,
            line=dict(width=0),
        ),
        hovertemplate=f"{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"{x_col} vs {y_col}  (r = {corr_val:.3f})",
            font=dict(size=14, color="#ddd"),
        ),
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    style_fig(fig, 380)
    st.plotly_chart(fig, use_container_width=True)


def _multi_boxplot(num_df: pd.DataFrame) -> None:
    cols_to_show = [c for c in num_df.columns if c not in ("latitude", "longitude")][:8]
    fig = go.Figure()
    fill_palette = [
        "rgba(108,99,255,0.12)", "rgba(0,210,255,0.12)", "rgba(124,77,255,0.12)",
        "rgba(255,107,107,0.12)", "rgba(255,217,61,0.12)", "rgba(107,203,119,0.12)",
        "rgba(77,150,255,0.12)", "rgba(255,107,157,0.12)",
    ]
    for i, col in enumerate(cols_to_show):
        data = num_df[col].dropna()
        color = ACCENT_SEQ[i % len(ACCENT_SEQ)]
        fig.add_trace(go.Box(
            y=data,
            name=col,
            marker=dict(color=color, size=2, opacity=0.3),
            line=dict(color=color),
            fillcolor=fill_palette[i % len(fill_palette)],
            boxmean=True,
        ))
    fig.update_layout(
        title=dict(text="Feature Distributions — All Numerical Columns", font=dict(size=14, color="#ddd")),
        yaxis_title="Value",
        showlegend=False,
    )
    style_fig(fig, 420)
    st.plotly_chart(fig, use_container_width=True)
