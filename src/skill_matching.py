"""Skill extraction and matching."""

from __future__ import annotations

import re

from src.utils import split_skills


COMMON_SKILLS = {
    "python",
    "sql",
    "machine learning",
    "deep learning",
    "nlp",
    "natural language processing",
    "pandas",
    "numpy",
    "scikit-learn",
    "sklearn",
    "tensorflow",
    "pytorch",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "fastapi",
    "streamlit",
    "spark",
    "airflow",
    "faiss",
    "vector search",
    "rag",
    "llm",
    "data engineering",
    "statistics",
    "matplotlib",
    "power bi",
    "tableau",
    "excel",
    "llms",
    "langchain",
    "redis",
    "terraform",
    "ci/cd",
    "cicd",
    "mlops",
    "hugging face",
    "vector databases",
    "postgres",
    "mongodb",
    "linux",
    "git",
}


def extract_required_skills(job_description: str, explicit_skills: list[str] | None = None) -> list[str]:
    """Extract required skills from explicit data or from a job description."""
    skills = set(split_skills(explicit_skills or []))
    text = job_description.lower()
    for skill in COMMON_SKILLS:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text):
            skills.add(skill.lower())
    return sorted(skills)


def calculate_skill_match(required_skills: list[str], candidate_skills: object) -> tuple[float, list[str], list[str]]:
    """Return skill match ratio, matched skills, and missing skills."""
    required = {skill.lower().strip() for skill in required_skills if skill.strip()}
    candidate = {skill.lower().strip() for skill in split_skills(candidate_skills)}
    if not required:
        return 1.0, [], []
    matched = sorted(
        skill for skill in required if skill in candidate or any(skill in item for item in candidate)
    )
    missing = sorted(required - set(matched))
    return len(matched) / len(required), matched, missing
