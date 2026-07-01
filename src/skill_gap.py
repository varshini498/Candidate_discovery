"""Skill gap analysis between job requirements and candidate profiles."""

from __future__ import annotations

import pandas as pd

from src.utils import split_skills


NICE_TO_HAVE_POOL = [
    "terraform",
    "redis",
    "kubernetes",
    "langchain",
    "mlops",
    "ci/cd",
    "monitoring",
    "pinecone",
]


def analyze_skill_gap(required_skills: list[str], candidate: pd.Series) -> dict[str, object]:
    """Compare required skills against candidate skills and explain impact."""
    candidate_skills = {skill.lower() for skill in split_skills(candidate.get("Top_Skills"))}
    required = [skill.lower() for skill in required_skills]
    matched = [
        skill
        for skill in required
        if skill in candidate_skills or any(skill in item for item in candidate_skills)
    ]
    missing = [skill for skill in required if skill not in matched]
    nice_to_have = [
        skill
        for skill in NICE_TO_HAVE_POOL
        if skill in candidate_skills and skill not in matched
    ]
    coverage = len(matched) / len(required) if required else 1.0
    severity = _severity(coverage)
    return {
        "matched": matched,
        "missing": missing,
        "nice_to_have": nice_to_have,
        "coverage": coverage,
        "severity": severity,
        "learning_gap": _learning_gap(missing),
        "hiring_impact": _hiring_impact(severity),
    }


def _severity(coverage: float) -> str:
    if coverage >= 0.85:
        return "Minor"
    if coverage >= 0.6:
        return "Moderate"
    return "High"


def _learning_gap(missing: list[str]) -> str:
    if not missing:
        return "No major learning gap detected."
    return "Candidate may need onboarding in " + ", ".join(missing[:5]) + "."


def _hiring_impact(severity: str) -> str:
    if severity == "Minor":
        return "Candidate is suitable for hiring after minimal onboarding."
    if severity == "Moderate":
        return "Candidate can be considered if the team can support targeted ramp-up."
    return "Candidate has material gaps for this role and needs careful validation."

