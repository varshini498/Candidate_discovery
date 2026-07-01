"""Job-description understanding for the recruiter copilot."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.skill_matching import COMMON_SKILLS, extract_required_skills
from src.utils import extract_required_years


ROLE_PATTERNS = [
    r"(senior|lead|principal|staff|junior|mid[- ]level)?\s*([a-z ]*engineer)",
    r"(senior|lead|principal|staff|junior|mid[- ]level)?\s*([a-z ]*scientist)",
    r"(senior|lead|principal|staff|junior|mid[- ]level)?\s*([a-z ]*analyst)",
    r"(senior|lead|principal|staff|junior|mid[- ]level)?\s*([a-z ]*developer)",
]

EDUCATION_TERMS = [
    "phd",
    "master",
    "masters",
    "m.tech",
    "mtech",
    "b.tech",
    "btech",
    "bachelor",
    "degree",
    "computer science",
]

CERTIFICATION_TERMS = [
    "aws certified",
    "azure certified",
    "gcp certified",
    "kubernetes certified",
    "certification",
    "certified",
]


@dataclass(frozen=True)
class JobInsights:
    """Structured interpretation of a pasted job description."""

    role: str
    seniority: str
    required_years: float
    primary_skills: list[str]
    secondary_skills: list[str]
    education: str
    certifications: list[str]
    confidence: float


def analyze_job_description(job_description: str) -> JobInsights:
    """Extract role, skills, seniority, education, and confidence from text."""
    text = job_description.lower()
    role, seniority = _extract_role_and_seniority(text)
    primary_skills = extract_required_skills(job_description)
    secondary_skills = _secondary_skills(text, primary_skills)
    education = _extract_education(text)
    certifications = _extract_certifications(text)
    years = extract_required_years(job_description) or _seniority_years(seniority)
    confidence = _confidence(role, primary_skills, years, education)
    return JobInsights(
        role=role,
        seniority=seniority,
        required_years=years,
        primary_skills=primary_skills,
        secondary_skills=secondary_skills,
        education=education,
        certifications=certifications,
        confidence=confidence,
    )


def _extract_role_and_seniority(text: str) -> tuple[str, str]:
    seniority = "Mid-Level"
    for keyword, label in [
        ("principal", "Principal"),
        ("staff", "Staff"),
        ("lead", "Lead"),
        ("senior", "Senior"),
        ("junior", "Junior"),
    ]:
        if keyword in text:
            seniority = label
            break

    role = "AI Engineer"
    for pattern in ROLE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            pieces = [part for part in match.groups() if part]
            role = " ".join(pieces).title()
            break
    if "llm" in text or "rag" in text:
        role = role.replace("Engineer", "AI Engineer") if "AI" not in role else role
    return role, seniority


def _secondary_skills(text: str, primary: list[str]) -> list[str]:
    primary_set = set(primary)
    nice_to_have_area = text.split("nice to have")[-1] if "nice to have" in text else text
    candidates = []
    for skill in sorted(COMMON_SKILLS):
        if skill in primary_set:
            continue
        if re.search(r"\b" + re.escape(skill) + r"\b", nice_to_have_area):
            candidates.append(skill)
    fallback = ["ci/cd", "langchain", "redis", "terraform", "mlops"]
    for skill in fallback:
        if skill not in primary_set and skill in text and skill not in candidates:
            candidates.append(skill)
    return candidates[:8]


def _extract_education(text: str) -> str:
    found = [term for term in EDUCATION_TERMS if term in text]
    if not found:
        return "Not specified"
    if "phd" in found:
        return "PhD preferred"
    if "master" in found or "masters" in found or "m.tech" in found or "mtech" in found:
        return "Masters preferred"
    return "Relevant degree preferred"


def _extract_certifications(text: str) -> list[str]:
    return [term.title() for term in CERTIFICATION_TERMS if term in text]


def _seniority_years(seniority: str) -> float:
    return {
        "Junior": 1.0,
        "Mid-Level": 3.0,
        "Senior": 5.0,
        "Lead": 7.0,
        "Staff": 8.0,
        "Principal": 10.0,
    }.get(seniority, 3.0)


def _confidence(role: str, skills: list[str], years: float, education: str) -> float:
    score = 0.45
    if role:
        score += 0.18
    if skills:
        score += min(len(skills) * 0.035, 0.22)
    if years:
        score += 0.1
    if education != "Not specified":
        score += 0.05
    return min(score, 0.98)

