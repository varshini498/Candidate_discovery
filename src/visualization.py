"""Visualization helpers for the Streamlit recruiter copilot."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pandas as pd


def radar_chart(candidate: pd.Series):
    """Create a radar chart for candidate score dimensions."""
    labels = ["Semantic", "Skills", "Experience", "Activity", "Education"]
    values = [
        float(candidate.get("Semantic_Score", 0)),
        float(candidate.get("Skill_Score", 0)),
        float(candidate.get("Experience_Score", 0)),
        float(candidate.get("Activity_Score", 0)),
        float(candidate.get("Education_Score", 0)),
    ]
    angles = [index / len(labels) * 2 * math.pi for index in range(len(labels))]
    values += values[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(4.5, 4.5), facecolor="#0f1218")
    axis = fig.add_subplot(111, polar=True)
    axis.set_facecolor("#151a23")
    axis.plot(angles, values, color="#6ee7b7", linewidth=2)
    axis.fill(angles, values, color="#6ee7b7", alpha=0.22)
    axis.set_xticks(angles[:-1])
    axis.set_xticklabels(labels, color="#e5e7eb")
    axis.set_yticks([0.25, 0.5, 0.75, 1.0])
    axis.set_yticklabels(["25", "50", "75", "100"], color="#9ca3af", fontsize=8)
    axis.grid(color="#374151", alpha=0.8)
    axis.spines["polar"].set_color("#374151")
    return fig


def score_color(score: float) -> str:
    """Return a color class key for a 0-100 score."""
    if score >= 80:
        return "green"
    if score >= 60:
        return "amber"
    if score >= 40:
        return "blue"
    return "red"

