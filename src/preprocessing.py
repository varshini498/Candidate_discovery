"""Text preprocessing helpers."""

from __future__ import annotations

import re
import string

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def preprocess_text(text: object) -> str:
    """Lowercase text, remove punctuation and stopwords, and normalize spaces."""
    if text is None:
        return ""
    value = str(text).lower()
    value = value.translate(str.maketrans("", "", string.punctuation))
    tokens = [token for token in value.split() if token not in ENGLISH_STOP_WORDS]
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def combine_candidate_text(
    resume: str = "",
    projects: str = "",
    skills: str = "",
    extra: str = "",
) -> str:
    """Build a single searchable profile text for semantic matching."""
    return preprocess_text(" ".join([resume, projects, skills, extra]))

