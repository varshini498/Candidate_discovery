"""Command-line entry point for intelligent candidate discovery."""

from __future__ import annotations

import argparse
import sys

from src.ranking import run_pipeline
from src.utils import configure_logging, export_rankings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank candidates for a job description.")
    parser.add_argument("--data-dir", default="data", help="Folder containing input data files.")
    parser.add_argument("--output", default="outputs/ranked_candidates.xlsx", help="Excel output path.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top candidates to export.")
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    try:
        ranked = run_pipeline(args.data_dir, top_n=args.top_n)
        output_path = export_rankings(ranked, args.output)
        print(f"\nTop {len(ranked)} candidates:")
        print(
            ranked[
                [
                    "Rank",
                    "Candidate_ID",
                    "Candidate_Name",
                    "Semantic_Score",
                    "Skill_Score",
                    "Experience_Score",
                    "Activity_Score",
                    "Final_Score",
                ]
            ].to_string(index=False)
        )
        print(f"\nExported ranked candidates to {output_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

