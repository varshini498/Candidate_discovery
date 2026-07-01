"""Hiring recommendation and confidence scoring."""

from __future__ import annotations


DEFAULT_THRESHOLDS = {
    "strong_hire": 95.0,
    "good_match": 80.0,
    "potential_match": 65.0,
}


def hiring_decision(
    final_score: float,
    semantic_score: float,
    skill_score: float,
    experience_score: float,
    activity_score: float,
    thresholds: dict[str, float] | None = None,
) -> dict[str, object]:
    """Return a recruiter-friendly decision, confidence, and explanation."""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    if final_score >= thresholds["strong_hire"]:
        decision = "Strong Hire"
    elif final_score >= thresholds["good_match"]:
        decision = "Good Match"
    elif final_score >= thresholds["potential_match"]:
        decision = "Potential Match"
    else:
        decision = "Not Recommended"

    confidence = _confidence(
        final_score,
        semantic_score,
        skill_score,
        experience_score,
        activity_score,
    )
    reasons = _decision_reasons(
        semantic_score,
        skill_score,
        experience_score,
        activity_score,
    )
    return {
        "decision": decision,
        "confidence": confidence,
        "recommendation": f"{decision}. Confidence: {confidence:.0f}%.",
        "reason_bullets": reasons,
        "reason_text": " ".join(reasons),
    }


def _confidence(
    final_score: float,
    semantic_score: float,
    skill_score: float,
    experience_score: float,
    activity_score: float,
) -> float:
    completeness = sum(
        score > 0 for score in [semantic_score, skill_score, experience_score, activity_score]
    ) / 4
    confidence = final_score * 0.72 + completeness * 18 + min(skill_score, experience_score) * 10
    return max(45.0, min(confidence, 99.0))


def _decision_reasons(
    semantic_score: float,
    skill_score: float,
    experience_score: float,
    activity_score: float,
) -> list[str]:
    reasons = []
    if semantic_score >= 0.75:
        reasons.append("Excellent semantic alignment with the role.")
    elif semantic_score >= 0.5:
        reasons.append("Meaningful semantic alignment with the job description.")
    else:
        reasons.append("Semantic alignment is limited and should be validated.")

    if skill_score >= 0.85:
        reasons.append("Matches nearly all required skills.")
    elif skill_score >= 0.6:
        reasons.append("Matches several core skills with some gaps.")
    else:
        reasons.append("Important required skills are missing.")

    if experience_score >= 1:
        reasons.append("Experience meets or exceeds the requirement.")
    elif experience_score >= 0.75:
        reasons.append("Experience is close to the required level.")
    else:
        reasons.append("Experience is below the expected level.")

    if activity_score >= 0.7:
        reasons.append("Candidate profile shows strong activity signals.")
    elif activity_score >= 0.4:
        reasons.append("Candidate profile has moderate activity signals.")
    else:
        reasons.append("Recent candidate activity is limited.")
    return reasons

