"""End-to-end candidate ranking pipeline."""

from __future__ import annotations

import pandas as pd

from src.activity_score import activity_label, calculate_activity_score
from src.embedding import EmbeddingModel
from src.experience import calculate_experience_score
from src.explainability import build_reason
from src.hiring_decision import hiring_decision
from src.interview_questions import generate_interview_questions
from src.llm_analysis import JobInsights
from src.preprocessing import combine_candidate_text, preprocess_text
from src.resume_summary import generate_resume_summary
from src.skill_gap import analyze_skill_gap
from src.skill_matching import calculate_skill_match, extract_required_skills
from src.utils import (
    CANDIDATE_HINTS,
    choose_candidate_frame,
    choose_job_description,
    detect_columns,
    load_data_files,
    safe_text,
)

WEIGHTS = {
    "semantic": 0.45,
    "skills": 0.25,
    "experience": 0.15,
    "activity": 0.10,
    "education": 0.05,
}


def rank_candidates(
    candidate_df: pd.DataFrame,
    job_description: str,
    required_skills: list[str] | None = None,
    required_years: float = 0.0,
    top_n: int = 20,
    model: EmbeddingModel | None = None,
    job_insights: JobInsights | None = None,
) -> pd.DataFrame:
    """Rank candidates for a job description and return explainable scores."""
    if candidate_df.empty:
        raise ValueError("Candidate dataset is empty.")

    mapping = detect_columns(candidate_df, CANDIDATE_HINTS)
    model = model or EmbeddingModel()
    required_skills = required_skills or (
        job_insights.primary_skills if job_insights else []
    )
    required_skills = extract_required_skills(job_description, required_skills)
    required_education = job_insights.education if job_insights else "Not specified"
    job_text = preprocess_text(job_description)

    candidate_texts = []
    for _, row in candidate_df.iterrows():
        candidate_texts.append(
            combine_candidate_text(
                resume=safe_text(row.get(mapping.get("resume"))),
                projects=safe_text(row.get(mapping.get("projects"))),
                skills=safe_text(row.get(mapping.get("skills"))),
            )
        )

    semantic_scores = model.semantic_scores(job_text, candidate_texts)
    rows = []
    for idx, row in candidate_df.reset_index(drop=True).iterrows():
        skill_score, matched, missing = calculate_skill_match(
            required_skills, row.get(mapping.get("skills"))
        )
        experience_score, years = calculate_experience_score(
            row.get(mapping.get("experience")), required_years
        )
        education = safe_text(row.get(mapping.get("education"))) or "Not available"
        education_score = _education_score(education, required_education)
        activity = calculate_activity_score(row, mapping)
        semantic = float(semantic_scores[idx]) if idx < len(semantic_scores) else 0.0
        final = (
            WEIGHTS["semantic"] * semantic
            + WEIGHTS["skills"] * skill_score
            + WEIGHTS["experience"] * experience_score
            + WEIGHTS["activity"] * activity
            + WEIGHTS["education"] * education_score
        )
        final_percent = round(final * 100, 2)
        base_candidate = {
            "Candidate_ID": _field_or_index(row, mapping.get("id"), idx + 1),
            "Candidate_Name": _field_or_index(row, mapping.get("name"), f"Candidate {idx + 1}"),
            "Semantic_Score": round(semantic, 4),
            "Skill_Score": round(skill_score, 4),
            "Experience_Score": round(experience_score, 4),
            "Activity_Score": round(activity, 4),
            "Education_Score": round(education_score, 4),
            "Final_Score": final_percent,
            "Matched_Skills": ", ".join(matched),
            "Missing_Skills": ", ".join(missing),
            "Top_Skills": safe_text(row.get(mapping.get("skills"))),
            "Projects": safe_text(row.get(mapping.get("projects"))),
            "Certifications": safe_text(row.get(mapping.get("certifications"))),
            "Education": education,
            "Previous_Companies": safe_text(row.get(mapping.get("company"))),
            "Location": safe_text(row.get(mapping.get("location"))),
            "Availability": safe_text(row.get(mapping.get("availability"))),
            "Experience_Years": years,
            "Activity_Label": activity_label(activity),
            "Semantic_Reason": _semantic_reason(semantic),
            "Skill_Reason": _skill_reason(matched, missing, required_skills),
            "Experience_Reason": _experience_reason(years, required_years),
            "Activity_Reason": _activity_reason(activity),
            "Education_Reason": _education_reason(education_score, education, required_education),
        }
        decision = hiring_decision(
            final_percent,
            semantic,
            skill_score,
            experience_score,
            activity,
        )
        gap = analyze_skill_gap(required_skills, pd.Series(base_candidate))
        base_candidate["Hiring_Recommendation"] = decision["decision"]
        base_candidate["Hiring_Confidence"] = round(float(decision["confidence"]), 1)
        base_candidate["Recommendation"] = decision["recommendation"]
        base_candidate["Recommendation_Reason"] = decision["reason_text"]
        base_candidate["Decision_Reasons"] = "\n".join(
            f"- {reason}" for reason in decision["reason_bullets"]
        )
        base_candidate["Skill_Coverage"] = round(float(gap["coverage"]) * 100, 1)
        base_candidate["Nice_To_Have_Skills"] = ", ".join(gap["nice_to_have"])
        base_candidate["Learning_Gap"] = gap["learning_gap"]
        base_candidate["Gap_Severity"] = gap["severity"]
        base_candidate["Hiring_Impact"] = gap["hiring_impact"]
        base_candidate["AI_Resume_Summary"] = generate_resume_summary(
            pd.Series(base_candidate)
        )
        base_candidate["Interview_Questions"] = "\n".join(
            generate_interview_questions(
                job_description,
                pd.Series(base_candidate),
                gap["missing"],
            )
        )
        base_candidate["Reason"] = build_reason(
            semantic,
            skill_score,
            experience_score,
            years,
            required_years,
            activity,
            matched,
            missing,
        )
        rows.append(
            base_candidate
        )

    ranked = pd.DataFrame(rows).sort_values("Final_Score", ascending=False)
    ranked.insert(0, "Rank", range(1, len(ranked) + 1))
    return ranked.head(top_n).reset_index(drop=True)


def run_pipeline(data_dir: str = "data", top_n: int = 20) -> pd.DataFrame:
    """Load data, infer schema, rank candidates, and return top results."""
    frames = load_data_files(data_dir)
    candidate_df = choose_candidate_frame(frames)
    job_description, skills, years = choose_job_description(frames)
    return rank_candidates(candidate_df, job_description, skills, years, top_n=top_n)


def _field_or_index(row: pd.Series, column: str | None, fallback: object) -> object:
    if column and safe_text(row.get(column)):
        return row.get(column)
    return fallback


def _education_score(candidate_education: str, required_education: str) -> float:
    if required_education == "Not specified":
        return 0.85 if candidate_education != "Not available" else 0.5
    text = candidate_education.lower()
    requirement = required_education.lower()
    if "phd" in requirement:
        return 1.0 if "phd" in text else 0.75 if "master" in text or "m.tech" in text else 0.45
    if "master" in requirement:
        return 1.0 if any(term in text for term in ["master", "m.tech", "mtech", "phd"]) else 0.7
    if "degree" in requirement:
        return 1.0 if any(term in text for term in ["b.tech", "btech", "bachelor", "degree", "master", "phd"]) else 0.55
    return 0.8


def _semantic_reason(score: float) -> str:
    if score >= 0.75:
        return "Resume responsibilities closely match the job requirements."
    if score >= 0.5:
        return "Profile has meaningful overlap with the role."
    return "Profile language is only partially aligned with the job."


def _skill_reason(matched: list[str], missing: list[str], required: list[str]) -> str:
    if not required:
        return "No explicit skills were required, so skill fit is treated as neutral."
    return f"Matched {len(matched)} out of {len(required)} required skills."


def _experience_reason(years: float, required_years: float) -> str:
    if required_years <= 0:
        return f"Candidate reports {years:g} years of experience."
    return f"Candidate has {years:g} years while the role asks for {required_years:g}."


def _activity_reason(score: float) -> str:
    if score >= 0.75:
        return "Recent activity and profile signals indicate high engagement."
    if score >= 0.4:
        return "Profile activity is moderate."
    return "Limited recent activity signals were found."


def _education_reason(score: float, education: str, required_education: str) -> str:
    if required_education == "Not specified":
        return f"Education was not required; candidate lists {education}."
    if score >= 0.85:
        return f"Education aligns with requirement: {education}."
    return f"Education may not fully match requirement: {required_education}."
