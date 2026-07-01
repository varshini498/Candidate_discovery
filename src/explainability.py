"""Plain-language explanations for candidate scores."""

from __future__ import annotations


def build_reason(
    semantic_score: float,
    skill_score: float,
    experience_score: float,
    experience_years: float,
    required_years: float,
    activity_score: float,
    matched_skills: list[str],
    missing_skills: list[str],
) -> str:
    """Create a concise explanation for a ranking decision."""
    reasons = []
    if semantic_score >= 0.8:
        reasons.append("Excellent semantic match with the job responsibilities.")
    elif semantic_score >= 0.6:
        reasons.append("Strong semantic overlap with the job requirements.")
    else:
        reasons.append("Moderate semantic match with room for closer alignment.")

    if skill_score >= 0.9:
        reasons.append("Has nearly all required skills.")
    elif skill_score >= 0.6:
        reasons.append(f"Matches key skills: {', '.join(matched_skills[:5])}.")
    elif missing_skills:
        reasons.append(f"Missing important skills: {', '.join(missing_skills[:5])}.")

    if required_years <= 0:
        reasons.append(f"Reports {experience_years:g} years of experience.")
    elif experience_score >= 1:
        reasons.append("Experience meets or exceeds the requirement.")
    else:
        reasons.append(
            f"Experience is below requirement: {experience_years:g} of {required_years:g} years."
        )

    if activity_score >= 0.75:
        reasons.append("Recent profile activity indicates high engagement.")
    elif activity_score >= 0.4:
        reasons.append("Activity signals indicate moderate engagement.")
    else:
        reasons.append("Limited activity signals are available.")

    return " ".join(reasons)

