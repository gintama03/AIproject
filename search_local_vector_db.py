from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from rag_intent_classifier import classify_query_intent  # noqa: E402
from search_local_vector_db import search as search_vector_db  # noqa: E402


DEFAULT_QA = ROOT / "data" / "evaluation" / "question_answer_all_eval_ready.csv"
DEFAULT_RETRIEVAL = ROOT / "results" / "retrieval" / "retrieval_evaluation_results.csv"
DEFAULT_METADATA = ROOT / "data" / "vector_db" / "metadata.csv"
DEFAULT_VECTOR_DB = ROOT / "data" / "vector_db"
DEFAULT_OUT_DIR = ROOT / "work" / "answer_quality"


SYSTEM_PROMPT = """너는 건양대학교 인공지능학과 RAG 챗봇이다.
반드시 제공된 근거 문서 안에서만 답한다.
근거가 없는 내용은 추측하지 말고 "제공된 자료에서는 확인되지 않습니다"라고 말한다.
질문에 연도가 있으면 해당 연도의 근거를 우선 사용한다.
답변은 1) 핵심 결론 2) 필요한 세부 항목 3) 근거 출처 순서로 간결하게 작성한다.
수치, 학점, 학년, 학기, 과목명, 예외 조건은 원문 표현을 최대한 유지한다."""


USER_PROMPT_TEMPLATE = """아래 질문에 답해줘.

[질문]
{question}

[질문 의도]
{intent}

[검색된 근거]
{retrieved_context}

[공통 답변 규칙]
- 근거에 있는 사실만 사용한다.
- 질문에서 요구한 연도와 다른 연도 근거가 함께 있으면 질문 연도를 우선한다.
- 과목명, 학점, 학년, 학기, 졸업요건, 예외 조건은 빠뜨리지 않는다.
- 자료에 없는 내용은 없다고 분명히 말한다.
- 마지막 줄에 "근거: ..." 형식으로 사용한 출처 파일명을 적는다.

[질문 유형별 추가 규칙]
{extra_rules}"""


COMMON_EXTRA_RULES = """- 직접적인 답이 한 문장으로 없더라도, 검색된 근거 안의 표/목록/개요에서 확인되는 사실을 종합해 답한다.
- 단, 근거 문서 밖의 배경이나 이유는 새로 만들어내지 않는다."""

COMPARISON_RULES = """- 비교형 질문이다.
- 직접적인 비교표나 "차이점" 문장이 없어도, 검색된 여러 연도 근거에 포함된 학과명, 학점, 교과목, 교육과정 구성 차이를 비교해서 답한다.
- 가능하면 "이전/현재" 또는 "A연도/B연도" 형태의 짧은 비교 표로 정리한다.
- 어느 한쪽 연도 근거가 부족하면 확인 가능한 연도 내용과 확인되지 않는 부분을 분리해서 말한다."""

MISSING_OR_RENAMED_RULES = """- 존재 여부 또는 변경 여부를 묻는 질문이다.
- 질문한 과목/제도명이 근거에 없으면 "해당 명칭은 확인되지 않음"이라고 답한다.
- 대신 근거에서 확인되는 유사 과목명, 대체 가능성이 있는 과목명, 관련 과목명을 함께 제시한다.
- "수강 가능"처럼 행정 판단이 필요한 내용은 근거에 명시되어 있을 때만 확정적으로 말한다."""

CREDIT_RULES = """- 학점/졸업요건 질문이다.
- 총 졸업학점, 전공학점, 교양필수, 중점교양, KY Vision, 설계교과 등 숫자를 먼저 답한다.
- 단일전공, 복수전공, 연계전공, 융합전공, 부전공 기준이 함께 있으면 구분해서 답한다."""

COURSE_RULES = """- 교과목 질문이다.
- 과목명, 학년, 학기, 학점, 이론/실습, 비고, 교과목 개요를 근거에서 확인되는 범위 안에서 정리한다.
- 과목명이 연도별로 다르면 해당 연도 기준 과목명을 우선한다."""

WEB_INFO_RULES = """- 학과소개/진로/자격증/입학정보 질문이다.
- 교육과정 PDF보다 학과 웹 크롤링 자료가 더 직접적인 근거이면 웹 자료 내용을 우선한다.
- 공식 근거에 자격증명, 진출 분야, 입학 조건이 없으면 없다고 말하고, 확인되는 관련 역량만 제시한다."""


def read_csv_auto(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str).fillna("")
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str).fillna("")


def split_values(value: object) -> list[str]:
    return [part.strip() for part in re.split(r"[;,]", str(value or "")) if part.strip()]


def compact_whitespace(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def detect_extra_rules(question: str, qa_group: str) -> tuple[str, str]:
    normalized = re.sub(r"\s+", "", question.lower())
    intent = str(classify_query_intent(question).get("intent", "general"))
    rules = [COMMON_EXTRA_RULES]

    years = re.findall(r"20[0-9]{2}", question)
    comparison_terms = ("비교", "차이", "다른", "바뀐", "개편", "현재", "지금", "전년도", "과거", "이전")
    if len(set(years)) >= 2 or any(term in normalized for term in comparison_terms):
        rules.append(COMPARISON_RULES)

    missing_terms = ("있던", "지금도", "수강할수", "존재", "없", "변경", "대체", "바뀌")
    if any(term in normalized for term in missing_terms):
        rules.append(MISSING_OR_RENAMED_RULES)

    if intent == "graduation_credit" or any(term in normalized for term in ("학점", "졸업최소", "이수학점")):
        rules.append(CREDIT_RULES)

    if intent in {"exact_course_schedule", "course_description", "programming_data_course"}:
        rules.append(COURSE_RULES)

    if qa_group == "department_web" or intent in {"department_intro_history", "admission", "career_certificate"}:
        rules.append(WEB_INFO_RULES)

    return intent, "\n".join(rules)


def build_context(chunk_ids: list[str], metadata_by_id: dict[str, dict[str, str]]) -> tuple[str, str]:
    blocks: list[str] = []
    source_names: list[str] = []

    for order, chunk_id in enumerate(chunk_ids, 1):
        row = metadata_by_id.get(chunk_id)
        if not row:
            continue
        source = row.get("source_file", "") or row.get("url", "")
        title = compact_whitespace(row.get("title", ""))
        category = compact_whitespace(row.get("category", ""))
        year = compact_whitespace(row.get("year", ""))
        content = compact_whitespace(row.get("content", ""))
        source_names.append(source)
        blocks.append(
            "\n".join(
                [
                    f"[근거 {order}]",
                    f"- chunk_id: {chunk_id}",
                    f"- source: {source}",
                    f"- year: {year}",
                    f"- category: {category}",
                    f"- title: {title}",
                    f"- content: {content}",
                ]
            )
        )

    unique_sources = []
    for source in source_names:
        if source and source not in unique_sources:
            unique_sources.append(source)
    return "\n\n".join(blocks), ";".join(unique_sources)


def fallback_chunk_ids(row: pd.Series) -> list[str]:
    return split_values(row.get("top5_chunk_ids", ""))


def merge_chunk_ids(*groups: list[str], limit: int) -> list[str]:
    merged: list[str] = []
    for group in groups:
        for chunk_id in group:
            if chunk_id and chunk_id not in merged:
                merged.append(chunk_id)
            if len(merged) >= limit:
                return merged
    return merged


def needs_current_curriculum(question: str) -> bool:
    normalized = re.sub(r"\s+", "", question.lower())
    return any(term in normalized for term in ("지금", "현재", "최신", "개편", "바뀐", "바뀌", "변경"))


def search_chunk_ids(vector_db: Path, question: str, top_k: int, fallback: list[str]) -> list[str]:
    try:
        results = search_vector_db(vector_db, question, top_k)
    except Exception:
        return fallback
    base_ids = [str(result.get("chunk_id", "")).strip() for result in results if str(result.get("chunk_id", "")).strip()]

    extra_ids: list[str] = []
    if needs_current_curriculum(question):
        try:
            current_results = search_vector_db(vector_db, f"{question} 2025 인공지능학과 최신 교육과정", max(5, top_k // 2))
            extra_ids = [
                str(result.get("chunk_id", "")).strip()
                for result in current_results
                if str(result.get("source_file", "")).strip() == "2025_cr.pdf"
                and str(result.get("chunk_id", "")).strip()
            ]
        except Exception:
            extra_ids = []

    if extra_ids:
        # 현재/최신 비교 질문에서는 기본 검색 결과가 한 연도에 쏠릴 수 있어
        # 2025 근거가 들어갈 자리를 일부 확보한다.
        reserved_extra = extra_ids[:3]
        base_limit = max(0, top_k - len(reserved_extra))
        chunk_ids = merge_chunk_ids(base_ids[:base_limit], reserved_extra, fallback, limit=top_k)
    else:
        chunk_ids = merge_chunk_ids(base_ids, fallback, limit=top_k)
    return chunk_ids or fallback


def build_user_prompt(question: str, context: str, intent: str, extra_rules: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        question=question,
        intent=intent,
        retrieved_context=context,
        extra_rules=extra_rules,
    )


def write_jsonl(frame: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in frame.to_dict(orient="records"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_report(frame: pd.DataFrame, out_dir: Path, top_k: int) -> str:
    qa_group_counts = frame["qa_group"].value_counts().to_dict()
    intent_counts = frame["answer_intent"].value_counts().to_dict()
    source_hit_count = int(frame["source_hit_at_1"].astype(str).str.lower().eq("true").sum())
    total = len(frame)

    def bullet_map(values: dict[str, int]) -> str:
        return "\n".join(f"- {key}: {value}" for key, value in sorted(values.items()))

    return f"""# Answer Quality Pack Report v2

## Summary

- Total prompt rows: {total}
- Retrieval context top_k: {top_k}
- Top1 source hit rows: {source_hit_count}/{total}
- Output directory: `{out_dir}`

## Improvements

- 비교형/연도형 질문 전용 답변 규칙 추가
- 과목 존재 여부/명칭 변경 질문 전용 답변 규칙 추가
- 학점/졸업요건 질문 전용 답변 규칙 추가
- 교과목 질문 전용 답변 규칙 추가
- 학과소개/진로/입학정보 질문 전용 답변 규칙 추가
- 답변 생성 근거를 top5에서 top{top_k}로 확대

## Output Files

- `answer_prompt_dataset.csv`
- `answer_prompt_dataset.jsonl`
- `answer_generation_prompt_template.md`
- `answer_quality_pack_report.md`

## QA Group Counts

{bullet_map(qa_group_counts)}

## Answer Intent Counts

{bullet_map(intent_counts)}

## How To Use

1. `answer_prompt_dataset.csv` 또는 `.jsonl`의 `system_prompt`, `user_prompt`를 LLM API에 전달한다.
2. API 응답을 `generated_answer` 컬럼으로 저장한다.
3. `tools/evaluate_answer_quality.py`로 체크포인트 포함 여부를 평가한다.

## Note

`expected_answer`와 `check_points`는 평가 기준이며, 실제 답변 생성 프롬프트에는 넣지 않는다.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build answer-generation prompt dataset for Week13 RAG evaluation.")
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--vector-db", type=Path, default=DEFAULT_VECTOR_DB)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    qa = read_csv_auto(args.qa)
    retrieval = read_csv_auto(args.retrieval)
    metadata = read_csv_auto(args.metadata)
    metadata_by_id = {
        str(row.get("chunk_id", "")): {str(key): str(value) for key, value in row.items()}
        for row in metadata.to_dict(orient="records")
    }

    merged = qa.merge(retrieval, on="question_id", how="left", suffixes=("", "_retrieval"))
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        question = str(row.get("question", ""))
        fallback_ids = fallback_chunk_ids(row)
        chunk_ids = search_chunk_ids(args.vector_db, question, args.top_k, fallback_ids)
        context, retrieved_sources = build_context(chunk_ids, metadata_by_id)
        intent, extra_rules = detect_extra_rules(question, str(row.get("qa_group", "")))
        rows.append(
            {
                "question_id": row.get("question_id", ""),
                "qa_group": row.get("qa_group", ""),
                "year": row.get("year", ""),
                "question": question,
                "answer_intent": intent,
                "context_top_k": args.top_k,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": build_user_prompt(question, context, intent, extra_rules),
                "retrieved_context": context,
                "retrieved_sources": retrieved_sources,
                "retrieved_chunk_ids": ";".join(chunk_ids),
                "expected_source": row.get("expected_source", ""),
                "expected_answer": row.get("expected_answer", ""),
                "check_points": row.get("check_points", ""),
                "top1_source": row.get("top1_source", ""),
                "top1_title": row.get("top1_title", ""),
                "source_hit_at_1": row.get("source_hit_at_1", ""),
                "source_hit_at_5": row.get("source_hit_at_5", ""),
            }
        )

    out = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / "answer_prompt_dataset.csv", index=False, encoding="utf-8-sig")
    write_jsonl(out, args.out_dir / "answer_prompt_dataset.jsonl")

    template = f"""# RAG Answer Generation Prompt Template v2

## System Prompt

```text
{SYSTEM_PROMPT}
```

## User Prompt Template

```text
{USER_PROMPT_TEMPLATE}
```
"""
    (args.out_dir / "answer_generation_prompt_template.md").write_text(template, encoding="utf-8-sig", newline="\n")
    (args.out_dir / "answer_quality_pack_report.md").write_text(
        build_report(out, args.out_dir, args.top_k),
        encoding="utf-8-sig",
        newline="\n",
    )

    print(f"rows={len(out)}")
    print(f"top_k={args.top_k}")
    print(f"output={args.out_dir}")


if __name__ == "__main__":
    main()
