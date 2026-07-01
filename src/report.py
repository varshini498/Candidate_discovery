"""Downloadable report generation."""

from __future__ import annotations

from io import BytesIO

import pandas as pd


def build_pdf_report(ranked: pd.DataFrame, job_summary: dict[str, object]) -> bytes:
    """Build a small valid PDF report using only the standard library."""
    lines = [
        "Intelligent Candidate Discovery",
        "AI Recruiter Copilot Report",
        "",
        f"Role: {job_summary.get('role', 'Unknown')}",
        f"Seniority: {job_summary.get('seniority', 'Unknown')}",
        f"Required Experience: {job_summary.get('years', 'Unknown')} years",
        f"Confidence: {job_summary.get('confidence', 0):.0%}",
        "",
        "Top Candidates",
    ]
    for _, row in ranked.head(10).iterrows():
        lines.extend(
            [
                "",
                f"#{row['Rank']} {row['Candidate_Name']} - {row['Final_Score']:.1f}%",
                f"Recommendation: {row['Hiring_Recommendation']}",
                f"Confidence: {row.get('Hiring_Confidence', 0):.0f}%",
                f"AI Resume Summary: {row.get('AI_Resume_Summary', '')}",
                f"Skill Coverage: {row.get('Skill_Coverage', 0):.0f}%",
                f"Skills: {row.get('Matched_Skills', '')}",
                f"Missing: {row.get('Missing_Skills', '') or 'None'}",
                f"Decision: {row.get('Recommendation', '')}",
                f"Reason: {row.get('Recommendation_Reason', row['Reason'])}",
            ]
        )
    return _simple_pdf(lines)


def _simple_pdf(lines: list[str]) -> bytes:
    stream = "BT /F1 11 Tf 50 780 Td 14 TL "
    safe_lines = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in lines]
    stream += " T* ".join(f"({line}) Tj" for line in safe_lines)
    stream += " ET"
    objects = [
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
        (
            "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj"
        ),
        "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
        f"5 0 obj << /Length {len(stream.encode('latin-1', 'ignore'))} >> stream\n{stream}\nendstream endobj",
    ]
    output = BytesIO()
    output.write(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(output.tell())
        output.write(obj.encode("latin-1", "ignore") + b"\n")
    xref = output.tell()
    output.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        output.write(f"{offset:010d} 00000 n \n".encode())
    output.write(
        (
            f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF"
        ).encode()
    )
    return output.getvalue()
