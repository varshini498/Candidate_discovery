"""Session-aware recruiter chat assistant."""

from __future__ import annotations

import re

import pandas as pd

from src.interview_questions import generate_interview_questions


def answer_chat(
    question: str,
    ranked: pd.DataFrame,
    job_description: str,
) -> str:
    """Answer recruiter questions using current ranking results."""
    if ranked.empty:
        return "Run candidate analysis first, then I can answer questions about the ranking."
    text = question.lower().strip()
    if not text:
        return "Ask about rankings, candidate gaps, certifications, skills, summaries, or interviews."

    candidate = _candidate_from_question(text, ranked)
    if "top 3" in text or "recommend top" in text:
        names = ranked.head(3)["Candidate_Name"].astype(str).tolist()
        return "Top 3 recommended candidates: " + ", ".join(names) + "."
    if "compare" in text:
        return _compare_answer(ranked)
    if "ranked first" in text or "why is candidate 1" in text:
        top = ranked.iloc[0]
        return f"{top['Candidate_Name']} ranks first at {top['Final_Score']:.1f}%. {top['Recommendation']}"
    if "with" in text or "having" in text or "has" in text:
        skill_matches = _skill_matches(text, ranked)
        if skill_matches:
            return "Matching candidates: " + ", ".join(skill_matches) + "."
    if "lacks" in text or "missing" in text:
        skill = _last_keyword(text)
        matches = ranked[
            ranked["Missing_Skills"].astype(str).str.lower().str.contains(skill, na=False)
        ]["Candidate_Name"].astype(str).tolist()
        return (
            f"Candidates lacking {skill}: " + ", ".join(matches) + "."
            if matches
            else f"I do not see candidates explicitly missing {skill}."
        )
    if candidate is not None and "interview" in text:
        missing = _split(candidate.get("Missing_Skills"))
        questions = generate_interview_questions(job_description, candidate, missing)
        return "\n".join(f"{idx}. {item}" for idx, item in enumerate(questions, start=1))
    if candidate is not None and "summar" in text:
        return str(candidate.get("AI_Resume_Summary", "No summary available."))
    if candidate is not None or "explain" in text:
        target = candidate if candidate is not None else ranked.iloc[0]
        return f"{target['Candidate_Name']} scores {target['Final_Score']:.1f}%. {target['Recommendation']}"
    if "production" in text:
        matches = ranked[
            ranked["AI_Resume_Summary"].astype(str).str.lower().str.contains("production", na=False)
            | ranked["Projects"].astype(str).str.lower().str.contains("production|deployed", na=False)
        ]["Candidate_Name"].astype(str).tolist()
        return "Candidates with production evidence: " + ", ".join(matches) + "."
    return "I can help compare candidates, explain rankings, find skills, list gaps, or generate interview questions."


def _candidate_from_question(text: str, ranked: pd.DataFrame) -> pd.Series | None:
    number_match = re.search(r"candidate\s+(\d+)", text)
    if number_match:
        index = int(number_match.group(1)) - 1
        if 0 <= index < len(ranked):
            return ranked.iloc[index]
    for _, row in ranked.iterrows():
        name = str(row["Candidate_Name"]).lower()
        if name in text or name.split()[0] in text:
            return row
    return None


def _compare_answer(ranked: pd.DataFrame) -> str:
    if len(ranked) < 2:
        return "I need at least two candidates to compare."
    first = ranked.iloc[0]
    second = ranked.iloc[1]
    winner = first if first["Final_Score"] >= second["Final_Score"] else second
    return (
        f"{first['Candidate_Name']} scores {first['Final_Score']:.1f}% while "
        f"{second['Candidate_Name']} scores {second['Final_Score']:.1f}%. "
        f"I recommend {winner['Candidate_Name']} because their overall evidence is stronger."
    )


def _skill_matches(text: str, ranked: pd.DataFrame) -> list[str]:
    keywords = [word for word in re.findall(r"[a-zA-Z+#./-]+", text) if len(word) > 2]
    stop = {"show", "candidates", "having", "with", "who", "has", "have"}
    keywords = [word for word in keywords if word not in stop]
    matches = []
    for _, row in ranked.iterrows():
        haystack = " ".join(
            str(row.get(col, ""))
            for col in ["Top_Skills", "Certifications", "Projects", "AI_Resume_Summary"]
        ).lower()
        if any(keyword in haystack for keyword in keywords):
            matches.append(str(row["Candidate_Name"]))
    return matches


def _last_keyword(text: str) -> str:
    words = [word for word in re.findall(r"[a-zA-Z+#./-]+", text) if len(word) > 2]
    return words[-1] if words else "that skill"


def _split(value: object) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]
