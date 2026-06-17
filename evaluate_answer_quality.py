from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAG_EVAL = ROOT / "results" / "answer_quality" / "rag" / "answer_quality_results_generated_answer.csv"
DEFAULT_NO_RAG_EVAL = ROOT / "results" / "answer_quality" / "no_rag" / "answer_quality_results_generated_answer.csv"
DEFAULT_OUT_DIR = ROOT / "work" / "rag_vs_no_rag_comparison"


def read_csv_auto(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str).fillna("")
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str).fillna("")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def summarize(name: str, frame: pd.DataFrame) -> dict[str, object]:
    total = len(frame)
    return {
        "method": name,
        "total": total,
        "pass": int(frame["status"].eq("pass").sum()),
        "review": int(frame["status"].eq("review").sum()),
        "fail": int(frame["status"].eq("fail").sum()),
        "pass_rate": round(frame["status"].eq("pass").mean() * 100, 2) if total else 0,
        "avg_checkpoint_ratio": round(numeric(frame["checkpoint_ratio"]).mean() * 100, 2) if total else 0,
        "avg_source_ratio": round(numeric(frame["source_ratio"]).mean() * 100, 2) if total else 0,
        "zero_checkpoint_rows": int(numeric(frame["checkpoint_ratio"]).eq(0).sum()),
        "zero_source_rows": int(numeric(frame["source_ratio"]).eq(0).sum()),
    }


def status_score(status: str) -> int:
    return {"fail": 0, "review": 1, "pass": 2}.get(str(status).lower(), -1)


def build_report(summary: pd.DataFrame, detail: pd.DataFrame, rag_eval: Path, no_rag_eval: Path) -> str:
    rag_row = summary[summary["method"].eq("RAG")].iloc[0]
    no_rag_row = summary[summary["method"].eq("No RAG")].iloc[0]
    improved = int(detail["status_delta"].gt(0).sum())
    same = int(detail["status_delta"].eq(0).sum())
    worse = int(detail["status_delta"].lt(0).sum())
    checkpoint_gain = float(rag_row["avg_checkpoint_ratio"]) - float(no_rag_row["avg_checkpoint_ratio"])
    source_gain = float(rag_row["avg_source_ratio"]) - float(no_rag_row["avg_source_ratio"])

    lines = [
        "# RAG vs No-RAG Answer Quality Comparison",
        "",
        "## Input",
        "",
        f"- RAG evaluation: `{rag_eval}`",
        f"- No-RAG evaluation: `{no_rag_eval}`",
        "",
        "## Summary",
        "",
        "| Method | Total | PASS | REVIEW | FAIL | PASS rate | Avg. checkpoint | Avg. source |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['method']} | {row['total']} | {row['pass']} | {row['review']} | {row['fail']} | "
            f"{row['pass_rate']:.2f}% | {row['avg_checkpoint_ratio']:.2f}% | {row['avg_source_ratio']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Difference",
            "",
            f"- RAG checkpoint improvement: `{checkpoint_gain:.2f}%p`",
            f"- RAG source-grounding improvement: `{source_gain:.2f}%p`",
            f"- Rows improved by RAG: `{improved}`",
            f"- Rows unchanged: `{same}`",
            f"- Rows worse with RAG: `{worse}`",
            "",
            "## Interpretation",
            "",
            "RAG 사용 방식은 질문과 관련된 학과 문서를 검색한 뒤 답변을 생성하므로, "
            "학점, 교과목, 입학 정보처럼 문서 근거가 필요한 질문에서 일반 AI보다 안정적인 답변을 기대할 수 있다. "
            "No-RAG 방식은 질문만 보고 답하므로 그럴듯한 문장을 만들 수는 있지만, 학교 내부 문서에 기반한 세부 정보와 출처 제시 능력은 제한적이다.",
            "",
            "## Notable No-RAG Failures",
            "",
        ]
    )

    failures = detail[detail["no_rag_status"].eq("fail")].head(15)
    if failures.empty:
        lines.append("- No No-RAG failures.")
    else:
        for _, row in failures.iterrows():
            lines.append(
                f"- `{row['question_id']}` {row['question']} "
                f"(No-RAG checkpoint {row['no_rag_checkpoint_ratio']:.2f}, "
                f"RAG checkpoint {row['rag_checkpoint_ratio']:.2f})"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RAG and no-RAG answer quality evaluation results.")
    parser.add_argument("--rag-eval", type=Path, default=DEFAULT_RAG_EVAL)
    parser.add_argument("--no-rag-eval", type=Path, default=DEFAULT_NO_RAG_EVAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rag = read_csv_auto(args.rag_eval)
    no_rag = read_csv_auto(args.no_rag_eval)
    shared_question_ids = set(rag["question_id"]).intersection(set(no_rag["question_id"]))
    rag_scoped = rag[rag["question_id"].isin(shared_question_ids)].copy()
    no_rag_scoped = no_rag[no_rag["question_id"].isin(shared_question_ids)].copy()
    merged = no_rag.merge(
        rag_scoped,
        on="question_id",
        suffixes=("_no_rag", "_rag"),
        how="inner",
    )

    detail = pd.DataFrame(
        {
            "question_id": merged["question_id"],
            "qa_group": merged.get("qa_group_no_rag", ""),
            "year": merged.get("year_no_rag", ""),
            "question": merged.get("question_no_rag", ""),
            "no_rag_status": merged["status_no_rag"],
            "rag_status": merged["status_rag"],
            "no_rag_checkpoint_ratio": numeric(merged["checkpoint_ratio_no_rag"]),
            "rag_checkpoint_ratio": numeric(merged["checkpoint_ratio_rag"]),
            "checkpoint_delta": numeric(merged["checkpoint_ratio_rag"]) - numeric(merged["checkpoint_ratio_no_rag"]),
            "no_rag_source_ratio": numeric(merged["source_ratio_no_rag"]),
            "rag_source_ratio": numeric(merged["source_ratio_rag"]),
            "source_delta": numeric(merged["source_ratio_rag"]) - numeric(merged["source_ratio_no_rag"]),
            "status_delta": merged["status_rag"].map(status_score) - merged["status_no_rag"].map(status_score),
            "no_rag_missing_terms": merged.get("checkpoint_missing_terms_no_rag", ""),
            "rag_missing_terms": merged.get("checkpoint_missing_terms_rag", ""),
        }
    )

    summary = pd.DataFrame(
        [
            summarize("No RAG", no_rag_scoped),
            summarize("RAG", rag_scoped),
        ]
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "rag_vs_no_rag_summary.csv"
    detail_path = args.out_dir / "rag_vs_no_rag_detail.csv"
    report_path = args.out_dir / "rag_vs_no_rag_comparison_report.md"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    report_path.write_text(build_report(summary, detail, args.rag_eval, args.no_rag_eval), encoding="utf-8-sig")

    print(f"summary={summary_path}")
    print(f"detail={detail_path}")
    print(f"report={report_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
