"""Shared feature engineering + scoring.

The SAME code is used in training and in production. If these ever differ,
the model silently gets worse, so never copy-paste this logic elsewhere.
"""
import pandas as pd

WINDOW = 10  # seconds of history for rolling features
FEATURES = [
    "eye_gaze_angle", "audio_db", "tab_switches",
    "gaze_rolling_10s", "audio_rolling_10s",
]


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Expects one row per second, sorted by time, with columns
    eye_gaze_angle, audio_db, tab_switches (and optionally is_cheating)."""
    df = df.copy()
    # "Connection dropped" trap: carry last gaze forward, interpolate audio
    df["eye_gaze_angle"] = df["eye_gaze_angle"].ffill().bfill()
    df["audio_db"] = df["audio_db"].interpolate(limit_direction="both")
    df["tab_switches"] = df["tab_switches"].fillna(0)
    if "is_cheating" in df.columns:
        df["is_cheating"] = df["is_cheating"].fillna(0).astype(int)
    # Noise trap: sustained behaviour, not 1-second blips
    df["gaze_rolling_10s"] = df["eye_gaze_angle"].rolling(WINDOW, min_periods=WINDOW).mean()
    df["audio_rolling_10s"] = df["audio_db"].rolling(WINDOW, min_periods=WINDOW).max()
    return df.dropna(subset=FEATURES)


def score_readings(model, readings: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """readings: columns timestamp, eye_gaze_angle, audio_db, tab_switches.
    Returns one row per scorable second with probability + flag."""
    readings = readings.sort_values("timestamp").reset_index(drop=True)
    # Put on a strict 1-second grid so rolling windows really mean 10 seconds
    readings["timestamp"] = pd.to_datetime(readings["timestamp"]).dt.floor("s")
    readings = (readings.groupby("timestamp")
                .agg(eye_gaze_angle=("eye_gaze_angle", "mean"),
                     audio_db=("audio_db", "mean"),
                     tab_switches=("tab_switches", "sum"))
                .asfreq("s").reset_index())
    feats = clean_and_engineer(readings)
    if feats.empty:
        return feats.assign(probability=[], flagged=[])
    feats["probability"] = model.predict_proba(feats[FEATURES])[:, 1]
    feats["flagged"] = feats["probability"] >= threshold
    return feats
