"""Modern Streamlit AI Recruiter Copilot."""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from src.chat_assistant import answer_chat
from src.embedding import EmbeddingModel
from src.llm_analysis import JobInsights, analyze_job_description
from src.ranking import rank_candidates
from src.report import build_pdf_report
from src.utils import (
    choose_candidate_frame,
    dataframe_to_xlsx_bytes,
    detect_columns,
    load_data_files,
    safe_text,
)
from src.visualization import radar_chart, score_color


st.set_page_config(
    page_title="Intelligent Candidate Discovery",
    page_icon="ID",
    layout="wide",
)


def inject_styles() -> None:
    """Apply a polished dark product theme."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #090d14;
            --panel: #111827;
            --panel-2: #151c2b;
            --text: #f8fafc;
            --muted: #9ca3af;
            --line: #243044;
            --green: #34d399;
            --red: #fb7185;
            --amber: #fbbf24;
            --blue: #60a5fa;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(52, 211, 153, 0.12), transparent 32rem),
                linear-gradient(135deg, #090d14 0%, #0c111d 48%, #111827 100%);
            color: var(--text);
        }
        [data-testid="stSidebar"] {
            background: #0f172a;
            border-right: 1px solid var(--line);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1440px;
        }
        h1, h2, h3 { letter-spacing: 0; }
        .hero {
            border: 1px solid var(--line);
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.96), rgba(21, 28, 43, 0.88));
            border-radius: 18px;
            padding: 1.4rem 1.6rem;
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
            margin-bottom: 1rem;
        }
        .eyebrow {
            color: var(--green);
            text-transform: uppercase;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: .08rem;
            margin-bottom: .35rem;
        }
        .subtitle {
            color: var(--muted);
            font-size: 1.05rem;
            margin-top: -.55rem;
        }
        .metric-card, .ai-card, .candidate-shell, .compare-card {
            border: 1px solid var(--line);
            background: rgba(17, 24, 39, 0.82);
            border-radius: 16px;
            padding: 1rem;
            height: 100%;
        }
        .metric-label { color: var(--muted); font-size: .82rem; }
        .metric-value { color: var(--text); font-size: 1.65rem; font-weight: 800; margin-top: .2rem; }
        .badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: .24rem .58rem;
            margin: .14rem .18rem .14rem 0;
            font-size: .78rem;
            font-weight: 700;
            border: 1px solid transparent;
        }
        .badge-green { color: #d1fae5; background: rgba(16, 185, 129, .16); border-color: rgba(52, 211, 153, .32); }
        .badge-red { color: #ffe4e6; background: rgba(244, 63, 94, .15); border-color: rgba(251, 113, 133, .32); }
        .badge-blue { color: #dbeafe; background: rgba(59, 130, 246, .15); border-color: rgba(96, 165, 250, .32); }
        .badge-amber { color: #fef3c7; background: rgba(245, 158, 11, .15); border-color: rgba(251, 191, 36, .34); }
        .rec-strong { color: var(--green); font-weight: 900; }
        .rec-good { color: var(--blue); font-weight: 900; }
        .rec-potential { color: var(--amber); font-weight: 900; }
        .rec-no { color: var(--red); font-weight: 900; }
        .small-muted { color: var(--muted); font-size: .88rem; }
        .score-pill {
            border-radius: 999px;
            padding: .3rem .7rem;
            background: rgba(52, 211, 153, .13);
            color: #a7f3d0;
            font-weight: 900;
            border: 1px solid rgba(52, 211, 153, .3);
        }
        div[data-testid="stExpander"] {
            border: 1px solid var(--line);
            background: rgba(17, 24, 39, .7);
            border-radius: 16px;
        }
        .stProgress > div > div > div > div { background-color: var(--green); }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def cached_load_data() -> dict[str, pd.DataFrame]:
    return load_data_files("data")


@st.cache_resource(show_spinner=False)
def cached_model(use_transformer: bool) -> EmbeddingModel:
    return EmbeddingModel(use_transformer=use_transformer)


@st.cache_data(show_spinner=False)
def cached_rank(
    candidate_df: pd.DataFrame,
    job_description: str,
    use_transformer: bool,
    top_n: int,
) -> tuple[pd.DataFrame, JobInsights, float]:
    start = time.perf_counter()
    insights = analyze_job_description(job_description)
    model = cached_model(use_transformer)
    ranked = rank_candidates(
        candidate_df,
        job_description,
        required_skills=insights.primary_skills,
        required_years=insights.required_years,
        top_n=top_n,
        model=model,
        job_insights=insights,
    )
    return ranked, insights, time.perf_counter() - start


def badge(text: object, color: str = "blue") -> str:
    return f'<span class="badge badge-{color}">{safe_text(text)}</span>'


def recommendation_class(value: str) -> str:
    return {
        "Strong Hire": "rec-strong",
        "Good Match": "rec-good",
        "Potential Match": "rec-potential",
        "Not Recommended": "rec-no",
    }.get(value, "rec-potential")


def score_bar(label: str, value: float) -> None:
    st.caption(f"{label}: {value * 100:.0f}%")
    st.progress(min(max(value, 0.0), 1.0))


def candidate_matches_filters(row: pd.Series, filters: dict[str, object]) -> bool:
    if row["Final_Score"] < filters["minimum_score"]:
        return False
    if filters["minimum_experience"] and row["Experience_Years"] < filters["minimum_experience"]:
        return False
    if filters["education"] != "Any" and filters["education"].lower() not in str(row["Education"]).lower():
        return False
    if filters["location"] and filters["location"].lower() not in str(row["Location"]).lower():
        return False
    if filters["availability"] != "Any" and filters["availability"].lower() not in str(row["Availability"]).lower():
        return False
    return True


def render_skill_visualization(required: list[str], matched_text: str) -> None:
    matched = {item.strip().lower() for item in matched_text.split(",") if item.strip()}
    chips = []
    for skill in required:
        is_matched = skill.lower() in matched
        suffix = "OK" if is_matched else "Missing"
        chips.append(badge(f"{skill} {suffix}", "green" if is_matched else "red"))
    st.markdown("".join(chips) or badge("No explicit skills detected", "amber"), unsafe_allow_html=True)


def render_chat_panel(ranked: pd.DataFrame, job_description: str) -> None:
    """Render session-aware recruiter chat."""
    st.markdown("### Recruiter Chat")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for message in st.session_state.chat_history[-6:]:
        role = "You" if message["role"] == "user" else "Copilot"
        st.markdown(f"**{role}:** {message['content']}")
    question = st.text_input(
        "Ask about candidates",
        placeholder="Recommend top 3 candidates",
        key="chat_question",
    )
    if st.button("Send", use_container_width=True):
        if question.strip():
            answer = answer_chat(question, ranked, job_description)
            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()


def compare_candidates(left: pd.Series, right: pd.Series) -> pd.DataFrame:
    rows = []
    for metric in [
        "Final_Score",
        "Semantic_Score",
        "Skill_Score",
        "Experience_Score",
        "Activity_Score",
        "Education_Score",
    ]:
        left_value = float(left[metric])
        right_value = float(right[metric])
        if metric != "Final_Score":
            left_value *= 100
            right_value *= 100
        better = left["Candidate_Name"] if left_value >= right_value else right["Candidate_Name"]
        rows.append(
            {
                "Signal": metric.replace("_", " "),
                str(left["Candidate_Name"]): round(left_value, 1),
                str(right["Candidate_Name"]): round(right_value, 1),
                "Better": better,
            }
        )
    return pd.DataFrame(rows)


inject_styles()

try:
    frames = cached_load_data()
    candidate_df = choose_candidate_frame(frames)
    mapping = detect_columns(candidate_df)
except Exception as exc:
    st.error(f"Could not load candidate data from the data folder: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("### Job Brief")
    job_description = st.text_area(
        "Paste job description",
        height=300,
        placeholder=(
            "We are looking for a Senior AI Engineer with experience in Python, "
            "LLMs, FastAPI, Docker, AWS, Vector Databases and RAG systems. "
            "Candidate should have 5+ years experience and production deployment experience."
        ),
        value=(
            "We are looking for a Senior AI Engineer with experience in Python, "
            "LLMs, FastAPI, Docker, AWS, Vector Databases and RAG systems. "
            "Candidate should have 5+ years experience and production deployment experience."
        ),
    )
    analyze = st.button("Analyze & Rank Candidates", type="primary", use_container_width=True)

    with st.expander("Optional filters", expanded=False):
        minimum_score = st.slider("Minimum score", 0, 100, 0)
        minimum_experience = st.slider("Minimum experience", 0, 15, 0)
        education_filter = st.selectbox("Education", ["Any", "B.Tech", "M.Tech", "M.S.", "PhD"])
        location_filter = st.text_input("Location")
        availability_filter = st.selectbox("Availability", ["Any", "Immediate", "15 days", "30 days", "60 days"])
        use_transformer = st.checkbox("Use Sentence Transformers", value=False)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">AI Recruiter Copilot</div>
      <h1>Intelligent Candidate Discovery</h1>
      <div class="subtitle">Paste a role brief. The copilot extracts hiring intent, ranks candidates, explains decisions, and prepares recruiter-ready reports.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if analyze or "ranked" in st.session_state:
    if analyze or "ranked" not in st.session_state:
        with st.spinner("Understanding the job and ranking candidates..."):
            ranked, insights, processing_time = cached_rank(
                candidate_df,
                job_description,
                use_transformer=use_transformer,
                top_n=max(100, len(candidate_df)),
            )
            st.session_state.ranked = ranked
            st.session_state.insights = insights
            st.session_state.processing_time = processing_time
    ranked = st.session_state.ranked
    insights = st.session_state.insights
    processing_time = st.session_state.processing_time

    filters = {
        "minimum_score": minimum_score,
        "minimum_experience": minimum_experience,
        "education": education_filter,
        "location": location_filter,
        "availability": availability_filter,
    }
    filtered = ranked[ranked.apply(lambda row: candidate_matches_filters(row, filters), axis=1)]

    top_score = float(ranked["Final_Score"].max()) if not ranked.empty else 0.0
    average_score = float(ranked["Final_Score"].mean()) if not ranked.empty else 0.0
    metric_cols = st.columns(5)
    metric_data = [
        ("Total Candidates", len(candidate_df)),
        ("Top Match", f"{top_score:.1f}%"),
        ("Average Match", f"{average_score:.1f}%"),
        ("Processing Time", f"{processing_time:.2f}s"),
        ("Confidence Score", f"{insights.confidence:.0%}"),
    ]
    for col, (label, value) in zip(metric_cols, metric_data):
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### AI Understanding")
    skill_badges = "".join(badge(skill, "green") for skill in insights.primary_skills)
    secondary_badges = "".join(badge(skill, "blue") for skill in insights.secondary_skills) or badge("None detected", "amber")
    cert_badges = "".join(badge(cert, "amber") for cert in insights.certifications) or badge("Not specified", "amber")
    st.markdown(
        f"""
        <div class="ai-card">
            <div class="small-muted">Role</div><h3>{insights.role}</h3>
            <p><b>Seniority:</b> {insights.seniority} &nbsp; <b>Required Experience:</b> {insights.required_years:g}+ years &nbsp; <b>Education:</b> {insights.education}</p>
            <p><b>Primary Skills</b><br>{skill_badges}</p>
            <p><b>Secondary Skills</b><br>{secondary_badges}</p>
            <p><b>Certifications</b><br>{cert_badges}</p>
            <span class="score-pill">Confidence {insights.confidence:.0%}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    search = st.text_input("Search candidates by name, skills, company, or education")
    if search:
        search_text = search.lower()
        searchable_cols = [
            "Candidate_Name",
            "Top_Skills",
            "Previous_Companies",
            "Education",
            "Location",
            "Certifications",
        ]
        mask = filtered[searchable_cols].astype(str).agg(" ".join, axis=1).str.lower().str.contains(search_text)
        filtered = filtered[mask]

    overview_col, chat_col = st.columns([0.68, 0.32])
    with overview_col:
        st.markdown("### Recruiter Shortlist")
        st.dataframe(
            filtered[
                [
                    "Rank",
                    "Candidate_Name",
                    "Final_Score",
                    "Hiring_Recommendation",
                    "Hiring_Confidence",
                    "Skill_Coverage",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )
    with chat_col:
        st.markdown('<div class="ai-card">', unsafe_allow_html=True)
        render_chat_panel(ranked, job_description)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Candidate Ranking")
    if filtered.empty:
        st.warning("No candidates match the current search or filters.")
    for _, candidate in filtered.iterrows():
        rec_class = recommendation_class(candidate["Hiring_Recommendation"])
        color = score_color(float(candidate["Final_Score"]))
        header = (
            f"#{candidate['Rank']} {candidate['Candidate_Name']} | "
            f"{candidate['Final_Score']:.1f}% | {candidate['Hiring_Recommendation']}"
        )
        with st.expander(header, expanded=int(candidate["Rank"]) == 1):
            top_left, top_right = st.columns([1.25, 0.75])
            with top_left:
                st.markdown(
                    f"""
                    <div class="candidate-shell">
                        <h3>{candidate['Candidate_Name']} <span class="score-pill">{candidate['Final_Score']:.1f}%</span></h3>
                        <p class="{rec_class}">{candidate['Hiring_Recommendation']} | Confidence {candidate['Hiring_Confidence']:.0f}%</p>
                        <p class="small-muted">{candidate['Location'] or 'Location not available'} | {candidate['Availability'] or 'Availability not available'}</p>
                        <p><b>AI Resume Summary</b><br>{candidate['AI_Resume_Summary']}</p>
                        <p><b>Hiring Decision</b><br>{candidate['Recommendation']} {candidate['Recommendation_Reason']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("**Decision Reasons**")
                st.text(candidate["Decision_Reasons"])
                st.markdown("**Resume Highlights**")
                st.markdown(f"**Top Skills:** {candidate['Top_Skills'] or 'Not available'}")
                st.markdown(f"**Projects:** {candidate['Projects'] or 'Not available'}")
                st.markdown(f"**Certifications:** {candidate['Certifications'] or 'Not available'}")
                st.markdown(f"**Education:** {candidate['Education']}")
                st.markdown(f"**Experience:** {candidate['Experience_Years']:g} years")
                st.markdown(f"**Previous Companies:** {candidate['Previous_Companies'] or 'Not available'}")
                st.markdown("**Skill Gap Analysis**")
                st.metric("Skill Coverage", f"{candidate['Skill_Coverage']:.0f}%")
                st.markdown(f"**Gap Severity:** {candidate['Gap_Severity']}")
                st.markdown(f"**Learning Gap:** {candidate['Learning_Gap']}")
                st.markdown(f"**Hiring Impact:** {candidate['Hiring_Impact']}")
                st.markdown("**Matched Skills**")
                matched = [item.strip() for item in str(candidate["Matched_Skills"]).split(",") if item.strip()]
                st.markdown("".join(badge(item, "green") for item in matched) or badge("No direct matches", "amber"), unsafe_allow_html=True)
                st.markdown("**Missing Skills**")
                missing = [item.strip() for item in str(candidate["Missing_Skills"]).split(",") if item.strip()]
                st.markdown("".join(badge(item, "red") for item in missing) or badge("No critical gaps", "green"), unsafe_allow_html=True)
                st.markdown("**Nice-to-have Skills**")
                nice = [item.strip() for item in str(candidate["Nice_To_Have_Skills"]).split(",") if item.strip()]
                st.markdown("".join(badge(item, "blue") for item in nice) or badge("None detected", "amber"), unsafe_allow_html=True)
                st.markdown("**Skill Match Visualization**")
                render_skill_visualization(insights.primary_skills, candidate["Matched_Skills"])
            with top_right:
                st.pyplot(radar_chart(candidate), use_container_width=True)
                score_bar("Overall Match", float(candidate["Final_Score"]) / 100)
                score_bar("Semantic", float(candidate["Semantic_Score"]))
                score_bar("Skills", float(candidate["Skill_Score"]))
                score_bar("Experience", float(candidate["Experience_Score"]))
                score_bar("Activity", float(candidate["Activity_Score"]))
                score_bar("Education", float(candidate["Education_Score"]))
            st.markdown("**Explainable AI Score Notes**")
            notes = pd.DataFrame(
                [
                    ("Semantic Match", f"{candidate['Semantic_Score'] * 100:.0f}%", candidate["Semantic_Reason"]),
                    ("Skill Match", f"{candidate['Skill_Score'] * 100:.0f}%", candidate["Skill_Reason"]),
                    ("Experience", f"{candidate['Experience_Score'] * 100:.0f}%", candidate["Experience_Reason"]),
                    ("Activity", f"{candidate['Activity_Score'] * 100:.0f}%", candidate["Activity_Reason"]),
                    ("Education", f"{candidate['Education_Score'] * 100:.0f}%", candidate["Education_Reason"]),
                ],
                columns=["Signal", "Score", "Reason"],
            )
            st.dataframe(notes, hide_index=True, use_container_width=True)
            with st.expander("Interview Questions", expanded=False):
                st.text_area(
                    "Copy questions",
                    value=candidate["Interview_Questions"],
                    height=240,
                    key=f"questions_{candidate['Candidate_ID']}",
                )

    st.markdown("### Candidate Comparison")
    names = ranked["Candidate_Name"].astype(str).tolist()
    compare_cols = st.columns(2)
    left_name = compare_cols[0].selectbox("Candidate A", names, index=0)
    right_name = compare_cols[1].selectbox("Candidate B", names, index=min(1, len(names) - 1))
    if left_name and right_name:
        left = ranked[ranked["Candidate_Name"].astype(str) == left_name].iloc[0]
        right = ranked[ranked["Candidate_Name"].astype(str) == right_name].iloc[0]
        comparison = compare_candidates(left, right)
        st.dataframe(comparison, hide_index=True, use_container_width=True)
        winner = left if left["Final_Score"] >= right["Final_Score"] else right
        st.success(
            f"AI recommendation: prioritize {winner['Candidate_Name']} for this role. "
            f"They have the stronger blended match at {winner['Final_Score']:.1f}%."
        )

    st.markdown("### Export")
    export_df = ranked[
        [
            "Rank",
            "Candidate_Name",
            "Final_Score",
            "Hiring_Recommendation",
            "Hiring_Confidence",
            "AI_Resume_Summary",
            "Skill_Coverage",
            "Matched_Skills",
            "Missing_Skills",
            "Nice_To_Have_Skills",
            "Gap_Severity",
            "Hiring_Impact",
            "Recommendation",
            "Recommendation_Reason",
            "Interview_Questions",
        ]
    ]
    export_cols = st.columns(2)
    export_cols[0].download_button(
        "Download Excel",
        data=dataframe_to_xlsx_bytes(export_df),
        file_name="ranked_candidates.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    export_cols[1].download_button(
        "Download PDF Report",
        data=build_pdf_report(
            ranked,
            {
                "role": insights.role,
                "seniority": insights.seniority,
                "years": insights.required_years,
                "confidence": insights.confidence,
            },
        ),
        file_name="candidate_discovery_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
else:
    st.markdown(
        """
        <div class="ai-card">
            <h3>Ready when you are</h3>
            <p class="small-muted">Paste a job description on the left and click Analyze & Rank Candidates. The copilot will infer role, seniority, skills, experience, education, and certifications automatically.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
