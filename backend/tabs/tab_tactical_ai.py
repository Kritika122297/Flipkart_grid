"""
tabs/tab_ai.py  —  AI Insights: RandomForest hotspot prediction + z-score anomaly detection
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from charts.utils import ACCENT_SEQ, PLOTLY_TEMPLATE, CHART_BG

_FEAT_COLS = [
    "hour", "day_num", "vehicle_size_score",
    "junction_factor", "near_junction", "is_rush_hour", "is_weekend",
]
_FEAT_LABELS = {
    "hour": "Hour of Day",
    "day_num": "Day of Week",
    "vehicle_size_score": "Vehicle Size Score",
    "junction_factor": "Junction Factor",
    "near_junction": "Near Junction",
    "is_rush_hour": "Rush Hour Flag",
    "is_weekend": "Weekend Flag",
}
_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ══════════════════════════════════════════════════════════════════════════════
#  CACHED ML FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _train_model(_df: pd.DataFrame):
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None, [], [], 0.0

    avail = [c for c in _FEAT_COLS if c in _df.columns]
    if len(avail) < 3:
        return None, [], [], 0.0

    X = _df[avail].fillna(0)
    y = (_df["cis"] > 50).astype(int)

    if len(X) > 200_000:
        idx = X.sample(200_000, random_state=42).index
        X, y = X.loc[idx], y.loc[idx]

    split = int(len(X) * 0.8)
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_tr, y_val = y.iloc[:split], y.iloc[split:]

    rf = RandomForestClassifier(
        n_estimators=150, max_depth=10, min_samples_leaf=20,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)

    try:
        auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
    except Exception:
        auc = 0.0

    imp = sorted(zip(avail, rf.feature_importances_), key=lambda x: x[1], reverse=True)
    return rf, avail, imp, round(float(auc), 3)


def _predict_day(_model, feats: list, dow: int, _df: pd.DataFrame):
    is_weekend = 1 if dow >= 5 else 0

    def _mean(col, default):
        return float(_df[col].mean()) if col in _df.columns else default

    defaults = {
        "vehicle_size_score": _mean("vehicle_size_score", 2.0),
        "junction_factor":    _mean("junction_factor",    0.5),
        "near_junction":      _mean("near_junction",      0.5),
    }

    rows = []
    for h in range(24):
        rows.append({
            "hour":               h,
            "day_num":            dow,
            "vehicle_size_score": defaults["vehicle_size_score"],
            "junction_factor":    defaults["junction_factor"],
            "near_junction":      defaults["near_junction"],
            "is_rush_hour":       int(7 <= h <= 10 or 16 <= h <= 20),
            "is_weekend":         is_weekend,
        })

    X = pd.DataFrame(rows)[feats]
    proba = _model.predict_proba(X)[:, 1] * 100
    levels = pd.cut(proba, bins=[0, 25, 50, 75, 100],
                    labels=["Low", "Medium", "High", "Critical"],
                    include_lowest=True)
    return pd.DataFrame({"hour": list(range(24)), "risk_score": proba,
                          "risk_level": levels.astype(str)})


@st.cache_data(show_spinner=False)
def _anomaly_detection(_df: pd.DataFrame):
    df2 = _df[["police_station", "created_datetime", "cis"]].copy()
    df2["date"] = pd.to_datetime(df2["created_datetime"], errors="coerce").dt.date
    df2 = df2.dropna(subset=["date"])

    daily = (
        df2.groupby(["police_station", "date"], observed=True)
        .agg(count=("cis", "size"), avg_cis=("cis", "mean"))
        .reset_index()
    )

    alerts = []
    for stn, grp in daily.groupby("police_station", observed=True):
        if len(grp) < 5:
            continue
        mu = grp["count"].mean()
        sigma = grp["count"].std()
        if sigma < 1:
            continue
        g = grp.copy()
        g["z_score"]  = (g["count"] - mu) / sigma
        g["expected"] = round(mu, 1)
        alerts.append(g[g["z_score"] > 2.0])

    if alerts:
        return pd.concat(alerts).sort_values("z_score", ascending=False).head(15)
    return pd.DataFrame(columns=["police_station", "date", "count", "avg_cis", "z_score", "expected"])


@st.cache_data(show_spinner=False)
def _station_hour_risk(_df: pd.DataFrame):
    if "police_station" not in _df.columns or "hour" not in _df.columns:
        return pd.DataFrame()
    top_stns = _df.groupby("police_station", observed=True)["cis"].mean().nlargest(12).index
    sub = _df[_df["police_station"].isin(top_stns)]
    return (
        sub.groupby(["police_station", "hour"], observed=True)["cis"]
        .mean()
        .unstack(fill_value=0)
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _risk_bar(pred_df: pd.DataFrame, day_name: str):
    cmap = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444", "Critical": "#7C3AED"}
    colors = [cmap.get(lv, "#6C63FF") for lv in pred_df["risk_level"]]

    fig = go.Figure(go.Bar(
        x=pred_df["hour"],
        y=pred_df["risk_score"],
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.0f}%" for v in pred_df["risk_score"]],
        textposition="outside",
        textfont=dict(size=9, color="#aaa"),
    ))
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(239,68,68,0.45)",
                  annotation_text="High-risk threshold",
                  annotation_font=dict(color="#888", size=10))
    fig.update_layout(
        title=dict(text=f"Predicted Congestion Risk — {day_name}",
                   font=dict(size=14, color="#E8E8E8")),
        template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        xaxis=dict(title="Hour of Day",
                   tickvals=list(range(0, 24, 2)),
                   ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)],
                   gridcolor="rgba(255,255,255,0.05)", color="#888"),
        yaxis=dict(title="Risk Score (%)", range=[0, 118],
                   gridcolor="rgba(255,255,255,0.05)", color="#888"),
        height=340, margin=dict(l=40, r=20, t=50, b=40), showlegend=False,
    )
    return fig


def _importance_bar(imp: list):
    labels = [_FEAT_LABELS.get(f, f) for f, _ in imp]
    values = [v * 100 for _, v in imp]
    palette = (ACCENT_SEQ * 4)[:len(imp)]

    fig = go.Figure(go.Bar(
        x=values[::-1], y=labels[::-1], orientation="h",
        marker=dict(color=palette[::-1], line=dict(width=0)),
        text=[f"{v:.1f}%" for v in values[::-1]],
        textposition="outside",
        textfont=dict(size=10, color="#aaa"),
    ))
    fig.update_layout(
        title=dict(text="What Drives High-Risk Violations?",
                   font=dict(size=14, color="#E8E8E8")),
        template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=11, color="#ccc")),
        height=280, margin=dict(l=10, r=70, t=50, b=20), showlegend=False,
    )
    return fig


def _risk_heatmap(pivot: pd.DataFrame):
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{c:02d}:00" for c in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[
            [0.0,  "#1a1a2e"],
            [0.3,  "#3B82F6"],
            [0.6,  "#F59E0B"],
            [0.85, "#EF4444"],
            [1.0,  "#7C3AED"],
        ],
        colorbar=dict(title=dict(text="Avg CIS", font=dict(color="#888")),
                      tickfont=dict(color="#888")),
        hovertemplate="Station: %{y}<br>Hour: %{x}<br>Avg CIS: %{z:.1f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Station × Hour Risk Matrix (Avg CIS)",
                   font=dict(size=14, color="#E8E8E8")),
        template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        xaxis=dict(tickfont=dict(size=8, color="#888")),
        yaxis=dict(tickfont=dict(size=9, color="#ccc")),
        height=max(300, 28 * len(pivot) + 80),
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig


def _anomaly_bar(anom: pd.DataFrame):
    fig = go.Figure(go.Bar(
        x=anom["z_score"].values[::-1],
        y=(anom["police_station"].astype(str) + " · " + anom["date"].astype(str)).values[::-1],
        orientation="h",
        marker=dict(
            color=anom["z_score"].values[::-1],
            colorscale=[[0, "#F59E0B"], [0.5, "#EF4444"], [1, "#7C3AED"]],
            line=dict(width=0),
        ),
        text=[f"z={z:.1f}  ({int(c)} violations)"
              for z, c in zip(anom["z_score"].values[::-1], anom["count"].values[::-1])],
        textposition="outside",
        textfont=dict(size=9, color="#aaa"),
    ))
    fig.update_layout(
        title=dict(text="Anomaly Severity — z-score vs station baseline",
                   font=dict(size=14, color="#E8E8E8")),
        template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=9, color="#ccc")),
        height=min(380, 44 + len(anom) * 28),
        margin=dict(l=10, r=100, t=50, b=20), showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN RENDER
# ══════════════════════════════════════════════════════════════════════════════

def render(df: pd.DataFrame) -> None:
    st.markdown(
        "<p class='section-header'>🤖 AI Insights</p>"
        "<p class='section-sub'>Machine-learning hotspot prediction, "
        "feature attribution, and statistical anomaly detection</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='pw-info-banner'>ℹ️ A <b>RandomForest classifier</b> is trained on your "
        "upload to predict high-CIS (>50) violation probability. "
        "<b>Anomaly detection</b> uses z-score statistics to flag stations with unusual daily spikes. "
        "Models retrain automatically on each new upload.</div>",
        unsafe_allow_html=True,
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    with st.spinner("🤖 Training RandomForest on your data…"):
        model, feats, imp, auc = _train_model(df)

    if model is None:
        st.markdown(
            "<div class='pw-warn-banner'>⚠️ scikit-learn not installed or too few feature columns. "
            "Run: <code>pip install scikit-learn</code></div>",
            unsafe_allow_html=True,
        )
        return

    # ── Model stat banner ──────────────────────────────────────────────────────
    n_train  = min(len(df), 200_000)
    high_pct = float((df["cis"] > 50).mean() * 100)
    st.markdown(
        f"""<div style="background:linear-gradient(135deg,rgba(0,210,100,0.09),
        rgba(108,99,255,0.09));border:1px solid rgba(0,210,100,0.22);
        border-radius:14px;padding:16px 26px;margin-bottom:22px;
        display:flex;gap:44px;flex-wrap:wrap;align-items:center;">
          <div><div style="color:#888;font-size:0.7rem;text-transform:uppercase;
          letter-spacing:.08em;">Model</div>
          <div style="color:#00D264;font-size:1.05rem;font-weight:800;">
          RandomForest Classifier</div></div>
          <div><div style="color:#888;font-size:0.7rem;text-transform:uppercase;
          letter-spacing:.08em;">Training Rows</div>
          <div style="color:#E8E8E8;font-size:1.05rem;font-weight:800;">
          {n_train:,}</div></div>
          <div><div style="color:#888;font-size:0.7rem;text-transform:uppercase;
          letter-spacing:.08em;">Validation AUC</div>
          <div style="color:#6C63FF;font-size:1.05rem;font-weight:800;">
          {auc:.3f}</div></div>
          <div><div style="color:#888;font-size:0.7rem;text-transform:uppercase;
          letter-spacing:.08em;">High-Risk Rate</div>
          <div style="color:#F59E0B;font-size:1.05rem;font-weight:800;">
          {high_pct:.1f}%</div></div>
          <div><div style="color:#888;font-size:0.7rem;text-transform:uppercase;
          letter-spacing:.08em;">Features Used</div>
          <div style="color:#E8E8E8;font-size:1.05rem;font-weight:800;">
          {len(feats)}</div></div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Section 1: Hourly risk forecast ───────────────────────────────────────
    st.markdown(
        "<p class='section-header' style='font-size:1.2rem;'>📈 Hourly Risk Forecast</p>",
        unsafe_allow_html=True,
    )

    sel_col, info_col = st.columns([1, 3])
    with sel_col:
        today_dow  = pd.Timestamp.now().dayofweek
        target_dow = st.selectbox(
            "Predict for day",
            options=list(range(7)),
            format_func=lambda i: _DAY_NAMES[i],
            index=(today_dow + 1) % 7,
            key="ai_dow_sel",
        )

    pred = _predict_day(model, feats, target_dow, df)
    peak_h  = int(pred.loc[pred["risk_score"].idxmax(), "hour"])
    peak_sc = float(pred["risk_score"].max())
    n_high  = int(pred["risk_level"].isin(["High", "Critical"]).sum())

    with info_col:
        st.markdown(
            f"<div style='display:flex;gap:32px;padding:14px 0 4px;flex-wrap:wrap;'>"
            f"<div><div style='color:#888;font-size:0.7rem;text-transform:uppercase;'>Peak Hour</div>"
            f"<div style='color:#EF4444;font-size:1.3rem;font-weight:800;'>{peak_h:02d}:00</div></div>"
            f"<div><div style='color:#888;font-size:0.7rem;text-transform:uppercase;'>Peak Risk</div>"
            f"<div style='color:#F59E0B;font-size:1.3rem;font-weight:800;'>{peak_sc:.0f}%</div></div>"
            f"<div><div style='color:#888;font-size:0.7rem;text-transform:uppercase;'>High-Risk Hours</div>"
            f"<div style='color:#7C3AED;font-size:1.3rem;font-weight:800;'>{n_high} / 24</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.plotly_chart(_risk_bar(pred, _DAY_NAMES[target_dow]), use_container_width=True)

    # ── Section 2: Feature importance + risk matrix ───────────────────────────
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    left, right = st.columns(2)

    with left:
        st.plotly_chart(_importance_bar(imp), use_container_width=True)

    with right:
        pivot = _station_hour_risk(df)
        if not pivot.empty:
            st.plotly_chart(_risk_heatmap(pivot), use_container_width=True)

    # ── Section 3: Anomaly detection ──────────────────────────────────────────
    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='section-header' style='font-size:1.2rem;'>🚨 Anomaly Detection</p>"
        "<p class='section-sub'>Stations with daily violation counts more than "
        "2 standard deviations above their own historical baseline</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Running z-score anomaly detection…"):
        anom = _anomaly_detection(df)

    if anom.empty:
        st.markdown(
            "<div class='pw-info-banner'>✅ No anomalies detected — "
            "violation patterns are statistically consistent across all stations.</div>",
            unsafe_allow_html=True,
        )
        return

    # Top-3 alert cards
    sev_colors = {0: "#7C3AED", 1: "#EF4444", 2: "#F59E0B"}
    cols = st.columns(min(3, len(anom)))
    for i, (_, row) in enumerate(anom.head(3).iterrows()):
        c = sev_colors.get(i, "#6C63FF")
        cols[i].markdown(
            f"<div style='background:rgba(30,33,48,0.65);border:1px solid {c}55;"
            f"border-left:4px solid {c};border-radius:12px;padding:15px;'>"
            f"<div style='color:{c};font-size:0.7rem;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px;'>"
            f"⚠️ Anomaly Detected</div>"
            f"<div style='color:#E8E8E8;font-weight:700;font-size:0.92rem;'>"
            f"{row['police_station']}</div>"
            f"<div style='color:#777;font-size:0.78rem;margin-top:3px;'>{row['date']}</div>"
            f"<div style='margin-top:10px;'>"
            f"<span style='color:{c};font-weight:800;font-size:1.15rem;'>{int(row['count'])}</span>"
            f"<span style='color:#666;font-size:0.78rem;'> violations</span>"
            f"</div>"
            f"<div style='color:#555;font-size:0.75rem;margin-top:3px;'>"
            f"Expected ≈ {row['expected']}  ·  "
            f"<b style='color:{c};'>z = {row['z_score']:.1f}σ</b></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    st.plotly_chart(_anomaly_bar(anom), use_container_width=True)


    st.markdown("#### 📋 Full Anomaly Table")
    disp = anom[["police_station", "date", "count", "expected", "avg_cis", "z_score"]].copy()
    disp.columns = ["Station", "Date", "Violations", "Expected", "Avg CIS", "Z-Score"]
    disp["Z-Score"] = disp["Z-Score"].round(2)
    disp["Avg CIS"] = disp["Avg CIS"].round(1)
    st.dataframe(disp, use_container_width=True, hide_index=True)
