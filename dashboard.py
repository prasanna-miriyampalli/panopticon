"""Demo dashboard.   Run:  streamlit run dashboard.py"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import score_readings

MODEL_DIR = Path(".")
model = joblib.load(MODEL_DIR / "model.joblib")
cfg = json.loads((MODEL_DIR / "config.json").read_text())

st.title("🛡️ Project Panopticon: Intelligent Proctoring")
threshold = st.sidebar.slider("Decision threshold", 0.50, 0.99, float(cfg["threshold"]), 0.01)
st.sidebar.caption("Higher = fewer false accusations, more missed cheaters.")


def simulate(kind, n=60, seed=1):
    rng = np.random.default_rng(seed)
    gaze, audio, tabs = rng.normal(8, 2, n), rng.normal(37, 3, n), np.zeros(n)
    if kind == "Sneeze (1 second)":
        gaze[30], audio[30] = 45, 85
    elif kind == "Cheating (sustained)":
        gaze[20:45] = rng.normal(42, 5, 25); audio[20:45] = rng.normal(65, 4, 25)
        tabs[[24, 33]] = 2
    ts = pd.date_range("2024-01-01 09:00:00", periods=n, freq="s")
    return pd.DataFrame({"timestamp": ts, "eye_gaze_angle": gaze,
                         "audio_db": audio, "tab_switches": tabs})


mode = st.radio("Input", ["Normal behaviour", "Sneeze (1 second)",
                          "Cheating (sustained)", "Upload CSV"], horizontal=True)
if mode == "Upload CSV":
    up = st.file_uploader("CSV with timestamp, eye_gaze_angle, audio_db, tab_switches")
    if up is None:
        st.stop()
    raw = pd.read_csv(up, parse_dates=["timestamp"])
else:
    raw = simulate(mode)

scored = score_readings(model, raw, threshold)
c1, c2, c3 = st.columns(3)
c1.metric("Max probability", f"{scored['probability'].max():.0%}")
c2.metric("Flagged seconds", int(scored["flagged"].sum()))
c3.metric("Verdict", "🚨 Human review" if scored["flagged"].any() else "✅ Clear")

st.subheader("Cheating probability")
chart = scored.set_index("timestamp")[["probability"]].assign(threshold=threshold)
st.line_chart(chart)
st.subheader("Sensors (raw vs 10s rolling)")
st.line_chart(scored.set_index("timestamp")[
    ["eye_gaze_angle", "gaze_rolling_10s", "audio_db", "audio_rolling_10s"]])
