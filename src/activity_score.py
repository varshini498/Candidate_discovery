"""Candidate activity and engagement scoring."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.utils import safe_text


def calculate_activity_score(row: pd.Series, mapping: dict[str, str | None]) -> float:
    """Calculate a 0-1 activity score from available behavioral fields."""
    signals: list[float] = []

    completion_col = mapping.get("profile_completion")
    if completion_col:
        completion = _numeric(row.get(completion_col))
        if completion is not None:
            signals.append(min(completion / 100.0, 1.0) if completion > 1 else completion)

    recent_col = mapping.get("recent_login")
    if recent_col:
        signals.append(_recency_score(row.get(recent_col)))

    for key in ["certifications", "projects", "applications", "assessments", "badges"]:
        col = mapping.get(key)
        if not col:
            continue
        value = row.get(col)
        numeric = _numeric(value)
        if numeric is not None:
            signals.append(min(numeric / 10.0, 1.0))
        else:
            text = safe_text(value)
            signals.append(min(len([part for part in text.split(",") if part.strip()]) / 5.0, 1.0))

    if not signals:
        return 0.5
    return float(max(0.0, min(sum(signals) / len(signals), 1.0)))


def activity_label(score: float) -> str:
    """Convert numeric activity to a human-friendly label."""
    if score >= 0.75:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


def _numeric(value: object) -> float | None:
    try:
        if isinstance(value, (list, tuple, set)):
            return None
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _recency_score(value: object) -> float:
    text = safe_text(value)
    if not text:
        return 0.0
    try:
        dt = pd.to_datetime(text, errors="coerce")
        if pd.isna(dt):
            return 0.5
        if dt.tzinfo is None:
            dt = dt.tz_localize(timezone.utc)
        days = max((datetime.now(timezone.utc) - dt.to_pydatetime()).days, 0)
        if days <= 7:
            return 1.0
        if days <= 30:
            return 0.8
        if days <= 90:
            return 0.5
        return 0.2
    except Exception:
        return 0.5
