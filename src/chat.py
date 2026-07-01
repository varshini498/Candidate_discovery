"""Rule-based recruiter chat assistant."""

from __future__ import annotations

import pandas as pd


def answer_recruiter_question(question: str, ranked: pd.DataFrame) -> str:
    """Answer common recruiter questions using ranked candidate data."""
    if ranked.empty:
        return "I do not have ranked candidates yet. Run the analysis first."

    text = question.lower().strip()
    if not text:
        return "Ask me about rankings, missing skills, comparisons, or specific technologies."

    top = ranked.iloc[0]
    if "ranked first" in text or "top" in text or "best" in text:
        return (
            f"{top['Candidate_Name']} is ranked first with {top['Final_Score']:.1f}% "
            f"overall match. {top['Reason']}"
        )

    if "compare" in text and len(ranked) >= 2:
        first = ranked.iloc[0]
        second = ranked.iloc[1]
        winner = first if first["Final_Score"] >= second["Final_Score"] else second
        return (
            f"{first['Candidate_Name']} scores {first['Final_Score']:.1f}% and "
            f"{second['Candidate_Name']} scores {second['Final_Score']:.1f}%. "
            f"I would prioritize {winner['Candidate_Name']} because their blended "
            "semantic, skills, and experience scores are stronger."
        )

    if "not recommended" in text:
        weak = ranked.sort_values("Final_Score").iloc[0]
        return (
            f"{weak['Candidate_Name']} is the weakest fit at {weak['Final_Score']:.1f}%. "
            f"Main gaps: {weak.get('Missing_Skills', 'not enough matching evidence')}."
        )

    for _, row in ranked.iterrows():
        candidate_text = " ".join(
            [
                str(row.get("Candidate_Name", "")),
                str(row.get("Top_Skills", "")),
                str(row.get("Previous_Companies", "")),
                str(row.get("Education", "")),
                str(row.get("Certifications", "")),
            ]
        ).lower()
        if any(token in candidate_text for token in text.split() if len(token) > 2):
            return (
                f"{row['Candidate_Name']} looks relevant. Overall match is "
                f"{row['Final_Score']:.1f}%. {row['Reason']}"
            )

    return (
        "I could not find a direct match for that question. Try asking about a "
        "candidate name, a skill such as AWS or Kubernetes, or a comparison."
    )

