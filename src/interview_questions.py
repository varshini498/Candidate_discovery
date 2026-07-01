"""Interview question generation for ranked candidates."""

from __future__ import annotations

import pandas as pd


def generate_interview_questions(
    job_description: str,
    candidate: pd.Series,
    missing_skills: list[str],
    limit: int = 10,
) -> list[str]:
    """Generate technical, behavioral, design, and skill-gap questions."""
    skills_text = str(candidate.get("Top_Skills", "")).lower()
    questions = []
    questions.extend(_technical_questions(skills_text, job_description.lower()))
    questions.extend(
        [
            "Behavioral: Describe your most challenging production project and your role in it.",
            "Behavioral: Tell me about a time you had to debug a model or API issue under pressure.",
            "Leadership: How do you mentor junior engineers or improve team engineering practices?",
            "System Design: Design a scalable AI service from ingestion to monitoring.",
        ]
    )
    for skill in missing_skills[:3]:
        questions.append(
            f"Skill Gap: You have limited visible {skill} experience. How would you ramp up and apply it in production?"
        )
    if "rag" in job_description.lower() or "llm" in job_description.lower():
        questions.append("Technical: Explain Retrieval-Augmented Generation and common failure modes.")
    if "fastapi" in job_description.lower():
        questions.append("Technical: How would you optimize and secure a high-traffic FastAPI service?")
    return _dedupe(questions)[:limit]


def _technical_questions(skills_text: str, job_text: str) -> list[str]:
    questions = []
    if "faiss" in skills_text or "vector" in job_text:
        questions.append("Technical: Compare FAISS with managed vector databases for semantic search.")
    if "docker" in skills_text or "docker" in job_text:
        questions.append("Technical: How would you containerize and deploy an AI inference service?")
    if "aws" in skills_text or "aws" in job_text:
        questions.append("Technical: Which AWS services would you use for a production ML workflow?")
    if "kubernetes" in skills_text or "kubernetes" in job_text:
        questions.append("System Design: How would you deploy a scalable AI application on Kubernetes?")
    if "python" in skills_text:
        questions.append("Technical: How do you profile and optimize Python data-processing code?")
    return questions


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    unique = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique

