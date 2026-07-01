"""Concise AI-style resume summaries."""

from __future__ import annotations

import pandas as pd

from src.utils import safe_text


def generate_resume_summary(candidate: pd.Series, max_words: int = 100) -> str:
    """Generate a compact professional summary from ranked candidate fields."""
    name = safe_text(candidate.get("Candidate_Name")) or "This candidate"
    years = candidate.get("Experience_Years", 0)
    skills = _top_items(candidate.get("Top_Skills"), limit=8)
    projects = safe_text(candidate.get("Projects"))
    education = safe_text(candidate.get("Education"))
    companies = safe_text(candidate.get("Previous_Companies"))

    domain = _infer_domain(skills, projects)
    leadership = _leadership_signal(projects, skills)
    sentences = [
        f"{name} is a {domain} professional with {years:g} years of experience.",
        f"Core strengths include {', '.join(skills) if skills else 'relevant technical skills'}.",
    ]
    if projects:
        sentences.append(f"Project experience includes {projects}.")
    if education and education != "Not available":
        sentences.append(f"Education background: {education}.")
    if companies:
        sentences.append(f"Previous experience spans {companies}.")
    if leadership:
        sentences.append(leadership)
    return _limit_words(" ".join(sentences), max_words)


def _top_items(value: object, limit: int) -> list[str]:
    items = [item.strip() for item in safe_text(value).split(",") if item.strip()]
    return items[:limit]


def _infer_domain(skills: list[str], projects: str) -> str:
    text = " ".join(skills + [projects]).lower()
    if any(term in text for term in ["llm", "rag", "nlp", "vector", "faiss"]):
        return "AI and machine learning"
    if any(term in text for term in ["api", "docker", "kubernetes", "aws"]):
        return "backend and cloud engineering"
    if any(term in text for term in ["tableau", "power bi", "analytics"]):
        return "data analytics"
    return "technology"


def _leadership_signal(projects: str, skills: list[str]) -> str:
    text = " ".join([projects, *skills]).lower()
    if any(term in text for term in ["lead", "mentor", "architect", "deployed", "production"]):
        return "Shows ownership through production delivery and technical execution."
    return ""


def _limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(".,") + "."

