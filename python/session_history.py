"""
session_history.py
------------------
Saves and retrieves per-patient progress scores across sessions.
Used to compare current trial performance against the previous session.
"""

import json
import os
from datetime import datetime

HISTORY_PATH = os.path.join(os.path.dirname(__file__), "session_history.json")


def load_history() -> dict:
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return {}


def save_score(patient_id: str, exercise: int, progress_score: float) -> float | None:
    """
    Saves the progress score for this trial and returns the
    previous session's score for the same patient + exercise.
    Returns None if this is the first session.
    """
    history = load_history()
    key = str(patient_id)

    if key not in history:
        history[key] = []

    # Find last score for this specific exercise
    past = [s for s in history[key] if s["exercise"] == exercise]
    last_score = past[-1]["progress_score"] if past else None

    # Append new entry
    history[key].append({
        "date":           datetime.now().isoformat(),
        "exercise":       exercise,
        "progress_score": round(progress_score, 3)
    })

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    return last_score