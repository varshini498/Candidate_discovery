# Intelligent Candidate Discovery

AI-powered candidate ranking system for the Data & AI Challenge. The project ranks candidates for a job description using semantic similarity, skill overlap, experience fit, and activity signals.

## Project Overview

The system loads candidate datasets from `data/`, detects likely schema columns automatically, understands a pasted job description, embeds job and candidate profile text with Sentence Transformers or a fast local fallback, calculates explainable component scores, ranks the top candidates, and exports recruiter-ready reports.

## Architecture

- `src/utils.py`: data loading, schema detection, profiling, exports
- `src/preprocessing.py`: text normalization
- `src/embedding.py`: Sentence Transformer embeddings and FAISS helper
- `src/skill_matching.py`: required skill extraction and matching
- `src/experience.py`: experience scoring
- `src/activity_score.py`: behavioral signal scoring
- `src/ranking.py`: weighted ranking pipeline
- `src/explainability.py`: plain-language ranking reasons
- `src/llm_analysis.py`: automatic job understanding for role, seniority, skills, education, and certifications
- `src/visualization.py`: radar chart helpers
- `src/chat.py`: recruiter copilot chat responses
- `src/report.py`: downloadable PDF report generation
- `main.py`: command-line runner
- `app.py`: single-page AI Recruiter Copilot

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run

Add candidate data files to `data/`. Supported formats are CSV, JSON, JSONL, XLSX, and XLS.

```bash
python main.py
```

The command exports:

```text
outputs/ranked_candidates.xlsx
```

Run the dashboard:

```bash
python -m streamlit run app.py
```

Paste a job description and click **Analyze & Rank Candidates**. The app automatically extracts role, seniority, years of experience, primary skills, secondary skills, education, and certifications.

## Dataset Description

The loader accepts flexible column names. It attempts to detect fields such as:

- candidate id
- candidate name
- resume or profile summary
- skills
- projects
- experience
- recent login
- profile completion
- certifications
- applications
- assessments
- badges

If a separate job-description file exists in `data/`, the first detected job row is used. Otherwise, the command uses a default AI Engineer job description. In the Streamlit app, paste or edit the job description directly.

## Ranking Formula

Final score uses the challenge weights:

```text
Final Score = 0.50 * Semantic Similarity
            + 0.25 * Skill Match
            + 0.15 * Experience
            + 0.10 * Activity
```

Scores are exported with:

- `Rank`
- `Candidate_ID`
- `Candidate_Name`
- `Semantic_Score`
- `Skill_Score`
- `Experience_Score`
- `Activity_Score`
- `Final_Score`
- `Reason`

## Results

The top ranked candidates are shown in the terminal and exported to Excel. The Streamlit dashboard also supports candidate search, score breakdown, and Excel download.

The Streamlit copilot includes:

- AI understanding card
- expandable candidate recommendation cards
- resume highlights and missing-skill badges
- radar chart and progress bars
- candidate comparison
- recruiter chat assistant
- Excel and PDF downloads

## Future Work

- Add recruiter feedback loops for learning-to-rank
- Support multi-job batch ranking
- Add model monitoring and embedding cache persistence
- Add API endpoints with FastAPI
- Add richer skill ontology and synonym matching
