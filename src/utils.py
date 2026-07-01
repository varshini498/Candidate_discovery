"""Shared utilities for data loading, schema detection, and exports."""

from __future__ import annotations

import json
import logging
import re
import zipfile
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)


CANDIDATE_HINTS = {
    "id": ["candidate_id", "candidate id", "id", "user_id", "profile_id"],
    "name": ["candidate_name", "candidate name", "name", "full_name", "fullname"],
    "resume": ["resume", "summary", "profile", "bio", "description", "about"],
    "skills": ["skills", "skill", "technologies", "tech_stack", "competencies"],
    "projects": ["projects", "project", "portfolio", "work_samples"],
    "experience": ["experience", "years_experience", "years of experience", "yoe"],
    "recent_login": ["recent_login", "last_login", "last_active", "last_seen"],
    "profile_completion": ["profile_completion", "completion", "profile_score"],
    "certifications": ["certifications", "certificates", "credentials"],
    "applications": ["applications", "application_count", "jobs_applied"],
    "assessments": ["assessments", "assessment_score", "test_score"],
    "badges": ["badges", "achievements", "awards"],
    "education": ["education", "degree", "qualification", "university", "college"],
    "company": ["company", "companies", "previous_companies", "employer"],
    "location": ["location", "city", "country", "current_location"],
    "availability": ["availability", "notice_period", "joining", "available"],
}

JOB_HINTS = {
    "job_description": [
        "job_description",
        "description",
        "jd",
        "responsibilities",
        "requirements",
        "job details",
    ],
    "required_skills": ["required_skills", "skills", "must_have", "requirements"],
    "required_experience": [
        "required_experience",
        "min_experience",
        "experience",
        "years",
    ],
}


def configure_logging() -> None:
    """Configure a concise log format for command-line execution."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_data_files(data_dir: str | Path = "data") -> dict[str, pd.DataFrame]:
    """Load all CSV, JSON, JSONL, and Excel files from a directory."""
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    supported = [*path.glob("*.csv"), *path.glob("*.json"), *path.glob("*.jsonl")]
    supported.extend(path.glob("*.xlsx"))
    supported.extend(path.glob("*.xls"))

    frames: dict[str, pd.DataFrame] = {}
    for file_path in sorted(supported):
        try:
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() == ".jsonl":
                df = pd.read_json(file_path, lines=True)
            elif file_path.suffix.lower() == ".json":
                df = _read_json(file_path)
            else:
                df = pd.read_excel(file_path)
            frames[file_path.name] = df
            LOGGER.info("Loaded %s with %s rows.", file_path.name, len(df))
            print_dataset_profile(file_path.name, df)
        except Exception as exc:  # pragma: no cover - defensive for unknown files
            LOGGER.warning("Skipping %s: %s", file_path.name, exc)
    return frames


def _read_json(file_path: Path) -> pd.DataFrame:
    with file_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                return pd.DataFrame(value)
        return pd.json_normalize(payload)
    raise ValueError("JSON root must be an object or array.")


def print_dataset_profile(name: str, df: pd.DataFrame) -> None:
    """Print columns, missing values, and sample records for a dataframe."""
    print(f"\n=== {name} ===")
    print("Columns:")
    print(list(df.columns))
    print("\nMissing values:")
    print(df.isna().sum().to_string())
    print("\nSample records:")
    print(df.head(3).to_string(index=False))


def detect_columns(
    df: pd.DataFrame,
    hints: dict[str, list[str]] | None = None,
) -> dict[str, str | None]:
    """Map canonical field names to similar columns in an arbitrary schema."""
    hints = hints or CANDIDATE_HINTS
    normalized_columns = {_normalize_name(col): col for col in df.columns}
    mapping: dict[str, str | None] = {}

    for target, candidates in hints.items():
        best_col = None
        best_score = 0.0
        for hint in candidates:
            normalized_hint = _normalize_name(hint)
            for normalized_col, original_col in normalized_columns.items():
                score = _similarity(normalized_hint, normalized_col)
                if normalized_hint in normalized_col or normalized_col in normalized_hint:
                    score = max(score, 0.9)
                if score > best_score:
                    best_score = score
                    best_col = original_col
        mapping[target] = best_col if best_score >= 0.58 else None
    return mapping


def choose_candidate_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pick the dataframe most likely to contain candidate rows."""
    if not frames:
        raise FileNotFoundError("No data files found in the data folder.")
    best_name = ""
    best_df: pd.DataFrame | None = None
    best_score = -1
    for name, df in frames.items():
        mapping = detect_columns(df, CANDIDATE_HINTS)
        score = sum(value is not None for value in mapping.values())
        if "candidate" in name.lower():
            score += 2
        if score > best_score:
            best_name, best_df, best_score = name, df, score
    LOGGER.info("Using %s as candidate dataset.", best_name)
    if best_df is None:
        raise FileNotFoundError("Could not identify candidate data.")
    return best_df.copy()


def choose_job_description(
    frames: dict[str, pd.DataFrame],
    explicit_job_text: str | None = None,
) -> tuple[str, list[str], float]:
    """Find a job description and optional requirements from inputs."""
    if explicit_job_text:
        return explicit_job_text, [], extract_required_years(explicit_job_text) or 0.0

    for name, df in frames.items():
        if "job" not in name.lower() and "description" not in name.lower():
            continue
        mapping = detect_columns(df, JOB_HINTS)
        job_col = mapping.get("job_description")
        if job_col and not df.empty:
            row = df.iloc[0]
            job_text = safe_text(row.get(job_col))
            skills = split_skills(row.get(mapping.get("required_skills")))
            years = extract_required_years(
                row.get(mapping.get("required_experience"))
            ) or extract_required_years(job_text) or 0.0
            LOGGER.info("Using first row of %s as job description.", name)
            return job_text, skills, years

    default_job = (
        "Senior AI Engineer with Python, machine learning, NLP, vector search, "
        "data pipelines, model deployment, and production software experience. "
        "Requires 5+ years of relevant experience."
    )
    LOGGER.warning("No job description found. Using a default AI Engineer role.")
    return default_job, [], 5.0


def safe_text(value: object) -> str:
    """Convert arbitrary values to clean text."""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(safe_text(item) for item in value)
    if value is None or pd.isna(value):
        return ""
    return str(value)


def split_skills(value: object) -> list[str]:
    """Split a skill field into normalized skill tokens."""
    text = safe_text(value)
    if not text:
        return []
    parts = re.split(r"[,;|/\n]+", text)
    return [part.strip().lower() for part in parts if part.strip()]


def extract_required_years(value: object) -> float | None:
    """Extract a required years-of-experience number from text or numeric data."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple, set)):
        value = " ".join(safe_text(item) for item in value)
    if value is None or pd.isna(value):
        return None
    text = safe_text(value).lower()
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs|yoe)", text)
    if matches:
        return float(matches[0])
    numeric = re.findall(r"\d+(?:\.\d+)?", text)
    return float(numeric[0]) if numeric else None


def normalize_series(values: Iterable[object]) -> pd.Series:
    """Normalize a sequence to the 0-1 range."""
    series = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0)
    min_value = series.min()
    max_value = series.max()
    if max_value == min_value:
        return pd.Series([1.0 if max_value > 0 else 0.0] * len(series))
    return (series - min_value) / (max_value - min_value)


def export_rankings(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Export rankings to an Excel workbook."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_excel(path, index=False, engine="openpyxl")
    except (ImportError, ModuleNotFoundError):
        LOGGER.warning("openpyxl is not installed. Using built-in XLSX writer.")
        path.write_bytes(dataframe_to_xlsx_bytes(df))
    return path


def dataframe_to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    """Create a simple XLSX workbook without optional Excel dependencies."""
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml", _content_types_xml())
        workbook.writestr("_rels/.rels", _root_rels_xml())
        workbook.writestr("xl/workbook.xml", _workbook_xml())
        workbook.writestr("xl/_rels/workbook.xml.rels", _workbook_rels_xml())
        workbook.writestr("xl/worksheets/sheet1.xml", _worksheet_xml(df))
    return output.getvalue()


def _normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _similarity(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>"""


def _root_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""


def _workbook_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="Ranked Candidates" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""


def _workbook_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"""


def _worksheet_xml(df: pd.DataFrame) -> str:
    rows = [_xlsx_row(1, list(df.columns))]
    for row_number, (_, row) in enumerate(df.iterrows(), start=2):
        rows.append(_xlsx_row(row_number, row.tolist()))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(rows)}</sheetData>"
        "</worksheet>"
    )


def _xlsx_row(row_number: int, values: list[object]) -> str:
    cells = []
    for column_index, value in enumerate(values, start=1):
        cell_ref = f"{_column_letter(column_index)}{row_number}"
        if isinstance(value, (int, float)) and not pd.isna(value):
            cells.append(f'<c r="{cell_ref}"><v>{value}</v></c>')
        else:
            text = escape(safe_text(value))
            cells.append(f'<c r="{cell_ref}" t="inlineStr"><is><t>{text}</t></is></c>')
    return f'<row r="{row_number}">{"".join(cells)}</row>'


def _column_letter(index: int) -> str:
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters
