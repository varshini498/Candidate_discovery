"""Experience scoring."""

from __future__ import annotations

from src.utils import extract_required_years


def calculate_experience_score(candidate_experience: object, required_years: float) -> tuple[float, float]:
    """Score candidate experience against the requirement."""
    years = extract_required_years(candidate_experience) or 0.0
    if required_years <= 0:
        return 1.0, years
    return min(years / required_years, 1.0), years

