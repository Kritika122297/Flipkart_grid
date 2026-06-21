import io
import os
import streamlit as st
import pandas as pd
from data.helpers import parse_violations, get_max_severity, get_vehicle_size, compute_time_factor

DEFAULT_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "jan to may police violation_anonymized791b166.csv"),
    os.path.join(os.path.dirname(__file__), "..", "jan to may police violation_anonymized791b166.csv"),
    r"C:\Users\kriti\Downloads\jan to may police violation_anonymized791b166.csv",
]


@st.cache_data(show_spinner="⚙️ Processing records — this takes ~10 seconds…")
def load_and_process_data(source):
    """Returns (cleaned_df, cleaning_stats_dict).

    ``source`` may be a file-system path (str) or raw CSV bytes (bytes).
    Accepting bytes lets callers skip disk I/O entirely, which is required
    for thread-safe multi-user deployments.
    """
    if isinstance(source, bytes):
        df = pd.read_csv(io.BytesIO(source), low_memory=False)
    else:
        df = pd.read_csv(source, low_memory=False)
    raw_rows = len(df)

    # ── datetime ──
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce")
    bad_datetime = int(df["created_datetime"].isna().sum())
    df = df.dropna(subset=["created_datetime"])

    df["hour"] = df["created_datetime"].dt.hour
    df["day_of_week"] = df["created_datetime"].dt.day_name()
    df["day_num"] = df["created_datetime"].dt.dayofweek
    df["month"] = df["created_datetime"].dt.month
    df["month_name"] = df["created_datetime"].dt.month_name()
    df["is_rush_hour"] = df["hour"].apply(
        lambda h: 1 if (7 <= h <= 10) or (16 <= h <= 20) else 0
    )
    df["is_weekend"] = df["day_num"].apply(lambda d: 1 if d >= 5 else 0)

    # ── violation parsing ──
    df["violation_list"] = df["violation_type"].apply(parse_violations)
    df["num_violations"] = df["violation_list"].apply(len)

    # ── scoring ──
    df["violation_severity"] = df["violation_list"].apply(get_max_severity)
    df["vehicle_size_score"] = df["vehicle_type"].apply(get_vehicle_size)
    df["near_junction"] = df["junction_name"].apply(
        lambda j: 0 if pd.isna(j) or str(j).strip().upper() == "NO JUNCTION" else 1
    )
    df["junction_factor"] = df["near_junction"].apply(lambda x: 2.0 if x else 1.0)
    df["time_factor"] = df.apply(
        lambda r: compute_time_factor(r["hour"], r["is_weekend"]), axis=1
    )

    # ── CIS ──
    raw_cis = (
        df["violation_severity"]
        * df["vehicle_size_score"]
        * df["time_factor"]
        * df["junction_factor"]
    )
    cis_min, cis_max = raw_cis.min(), raw_cis.max()
    df["cis"] = (
        ((raw_cis - cis_min) / (cis_max - cis_min) * 100) if cis_max > cis_min else 50
    )

    # ── geo filter ──
    before_geo = len(df)
    df = df[
        (df["latitude"].between(12.5, 13.5)) & (df["longitude"].between(77.0, 78.0))
    ]
    bad_coords = before_geo - len(df)

    # ── missing value profile (raw columns only) ──
    raw_cols = [
        "id", "police_station", "location", "vehicle_type", "violation_type",
        "junction_name", "latitude", "longitude", "created_datetime",
    ]
    missing = {
        col: round(df[col].isna().mean() * 100, 1)
        for col in raw_cols if col in df.columns and df[col].isna().any()
    }

    stats = {
        "raw_rows": raw_rows,
        "dropped_bad_datetime": bad_datetime,
        "dropped_invalid_coords": bad_coords,
        "final_rows": len(df),
        "n_cols": len(df.columns),
        "date_min": df["created_datetime"].min().strftime("%d %b %Y"),
        "date_max": df["created_datetime"].max().strftime("%d %b %Y"),
        "date_range_days": (df["created_datetime"].max() - df["created_datetime"].min()).days,
        "missing": missing,
        "unique_stations": int(df["police_station"].nunique()),
        "unique_locations": int(df["location"].nunique()),
    }

    # ── Memory optimisation ───────────────────────────────────────────────────
    # Downcast int64 → smallest int and float64 → float32 (saves ~50-70 % RAM).
    for col in df.select_dtypes(include="int64").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include="float64").columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    # Convert high-cardinality repeating strings to category (big RAM win for
    # columns like police_station and vehicle_type that repeat thousands of times).
    _cat_cols = [
        "police_station", "vehicle_type", "violation_type",
        "junction_name", "location", "day_of_week", "month_name",
    ]
    for col in _cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df, stats


def auto_load():
    """Try default paths. Returns (df, stats) or (None, None)."""
    for p in DEFAULT_PATHS:
        if os.path.isfile(p):
            try:
                return load_and_process_data(p)
            except Exception as e:
                st.error(f"Error loading data: {e}")
    return None, None


def load_demo_data() -> str:
    """Generate synthetic BTP violation data and return the CSV path."""
    import numpy as np

    demo_path = os.path.join(os.path.dirname(__file__), "_demo_data.csv")
    if os.path.exists(demo_path):
        return demo_path

    rng = np.random.default_rng(42)
    n   = 1000

    stations = [
        "Koramangala", "Indiranagar", "Silk Board", "HSR Layout",
        "Whitefield", "Marathahalli", "Jayanagar", "Rajajinagar",
        "Electronic City", "Yelahanka",
    ]
    locations = [
        "80 Feet Road", "100 Feet Road", "Sarjapur Road", "Hosur Road",
        "Old Airport Road", "Outer Ring Road", "MG Road", "Brigade Road",
    ]
    vehicle_types   = ["Car", "Two-wheeler", "Bus", "Auto", "HGV/Truck", "Tanker"]
    violation_types = [
        "No Parking", "Double Parking", "Blocking Traffic",
        "Footpath Parking", "No Parking Zone",
    ]
    junctions = [
        "Sony World Signal", "Silk Board Junction", "Marathahalli Bridge",
        "KR Puram Signal", "NO JUNCTION",
    ]
    lat_lon = {
        "Koramangala":     (12.935, 77.624),
        "Indiranagar":     (12.978, 77.641),
        "Silk Board":      (12.917, 77.622),
        "HSR Layout":      (12.911, 77.637),
        "Whitefield":      (12.969, 77.750),
        "Marathahalli":    (12.959, 77.700),
        "Jayanagar":       (12.925, 77.583),
        "Rajajinagar":     (12.992, 77.553),
        "Electronic City": (12.844, 77.660),
        "Yelahanka":       (13.101, 77.594),
    }

    station_col = rng.choice(stations, size=n)
    lats = [lat_lon[s][0] + rng.normal(0, 0.008) for s in station_col]
    lons = [lat_lon[s][1] + rng.normal(0, 0.008) for s in station_col]

    start_ts   = pd.Timestamp("2024-01-01")
    end_ts     = pd.Timestamp("2024-05-31")
    total_secs = int((end_ts - start_ts).total_seconds())
    timestamps = [
        start_ts + pd.Timedelta(seconds=int(s))
        for s in rng.integers(0, total_secs, size=n)
    ]

    demo_df = pd.DataFrame({
        "id":               range(1, n + 1),
        "police_station":   station_col,
        "location":         rng.choice(locations, size=n),
        "vehicle_type":     rng.choice(
            vehicle_types, size=n, p=[0.30, 0.35, 0.10, 0.10, 0.10, 0.05]
        ),
        "violation_type":   rng.choice(violation_types, size=n),
        "junction_name":    rng.choice(junctions, size=n, p=[0.15, 0.15, 0.15, 0.15, 0.40]),
        "latitude":         lats,
        "longitude":        lons,
        "created_datetime": timestamps,
    })

    demo_df.to_csv(demo_path, index=False)
    return demo_path
