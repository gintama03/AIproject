from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "final_rag" / "final_rag_chunks.csv"
DEFAULT_OUTPUT = ROOT / "data" / "vector_db"

DEFAULT_DIMENSION = 4096
EMBEDDING_METHOD = "local_hash_tfidf_char_word_v1"


def stable_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def normalize_text(text: object) -> str:
    value = str(text or "").lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def token_features(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    compact = re.sub(r"\s+", "", text)
    features: list[str] = []

    for match in re.finditer(r"[0-9a-zA-Z가-힣]+", text):
        token = match.group(0)
        if len(token) >= 2:
            features.append("w:" + token)
        if re.fullmatch(r"[0-9]{4}", token):
            features.append("year:" + token)
        if re.fullmatch(r"[0-9]{5}[a-zA-Z]", token):
            features.append("code:" + token.upper())

    for n in (2, 3, 4):
        if len(compact) < n:
            continue
        for index in range(len(compact) - n + 1):
            features.append(f"c{n}:" + compact[index : index + n])

    return features


def row_text(row: pd.Series) -> str:
    # Metadata fields are repeated so short, high-signal labels can compete
    # with longer body text during local retrieval.
    parts = [
        row.get("title", ""),
        row.get("title", ""),
        row.get("title", ""),
        row.get("category", ""),
        row.get("category", ""),
        row.get("category", ""),
        row.get("category", ""),
        row.get("section", ""),
        row.get("section", ""),
        row.get("course_code", ""),
        row.get("course_code", ""),
        row.get("course_code", ""),
        row.get("course_name", ""),
        row.get("course_name", ""),
        row.get("course_name", ""),
        row.get("year", ""),
        row.get("year", ""),
        row.get("content", ""),
    ]
    return "\n".join(str(part) for part in parts if str(part).strip())


def build_vectors(rows: pd.DataFrame, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(rows), dimension), dtype=np.float32)
    df_counts = np.zeros(dimension, dtype=np.int32)

    row_feature_indices: list[list[int]] = []
    for _, row in rows.iterrows():
        features = token_features(row_text(row))
        indices = [stable_hash(feature) % dimension for feature in features]
        row_feature_indices.append(indices)
        for index in set(indices):
            df_counts[index] += 1

    doc_count = max(len(rows), 1)
    idf = (np.log((1.0 + doc_count) / (1.0 + df_counts)) + 1.0).astype(np.float32)

    for row_index, indices in enumerate(row_feature_indices):
        if not indices:
            continue
        counts: dict[int, int] = {}
        for index in indices:
            counts[index] = counts.get(index, 0) + 1
        for index, count in counts.items():
            matrix[row_index, index] = (1.0 + math.log(count)) * idf[index]

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = (matrix / norms).astype(np.float32)
    return matrix, idf


def write_jsonl(frame: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in frame.to_dict(orient="records"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_report(rows: pd.DataFrame, vectors: np.ndarray, output_dir: Path, dimension: int) -> str:
    category_counts = rows["category"].value_counts().to_dict()
    source_group_counts = rows["source_group"].value_counts().to_dict()
    needs_review = int(rows["needs_review"].astype(str).str.lower().eq("true").sum())
    empty_content = int(rows["content"].astype(str).str.strip().eq("").sum())
    duplicate_chunk_ids = int(rows["chunk_id"].duplicated().sum())
    replacement_rows = int(
        rows.astype(str).apply(lambda col: col.str.contains("\ufffd", regex=False)).any(axis=1).sum()
    )

    def bullet_map(values: dict[str, int]) -> str:
        return "\n".join(f"- {key}: {value}" for key, value in sorted(values.items()))

    return f"""# Local Vector DB Report

## Summary
- Status: usable
- Input file: `final_rag_chunks.csv`
- Output directory: `{output_dir.name}`
- Embedding method: `{EMBEDDING_METHOD}`
- Similarity: cosine
- Vector dimension: {dimension}
- Vector count: {vectors.shape[0]}
- Metadata rows: {len(rows)}

## Output Files
- `vectors.npz`
- `metadata.csv`
- `metadata.jsonl`
- `manifest.json`
- `vector_db_report.md`

## Quality Checks
- Duplicate chunk_id rows: {duplicate_chunk_ids}
- Empty content rows: {empty_content}
- Replacement character rows: {replacement_rows}
- needs_review rows: {needs_review}

## Source Group Counts
{bullet_map(source_group_counts)}

## Category Counts
{bullet_map(category_counts)}

## Note
This is a local reproducible vector index built without external model downloads.
For production semantic retrieval, the same `final_rag_chunks.csv` can later be embedded with OpenAI, SentenceTransformers, Chroma, FAISS, or Supabase Vector.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local vector DB from final RAG chunks.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    args = parser.parse_args()

    chunks = pd.read_csv(args.input, encoding="utf-8-sig", dtype=str).fillna("")
    required = {"chunk_id", "doc_id", "content", "title", "category", "source_group", "needs_review"}
    missing = sorted(required - set(chunks.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    args.output.mkdir(parents=True, exist_ok=True)

    vectors, idf = build_vectors(chunks, args.dimension)
    np.savez_compressed(
        args.output / "vectors.npz",
        vectors=vectors,
        idf=idf,
        dimension=np.array([args.dimension], dtype=np.int32),
    )

    chunks.to_csv(args.output / "metadata.csv", index=False, encoding="utf-8-sig")
    write_jsonl(chunks, args.output / "metadata.jsonl")

    manifest = {
        "embedding_method": EMBEDDING_METHOD,
        "similarity": "cosine",
        "dimension": args.dimension,
        "vector_count": int(vectors.shape[0]),
        "metadata_rows": int(len(chunks)),
        "input_file": str(args.input.relative_to(ROOT)),
        "files": {
            "vectors": "vectors.npz",
            "metadata_csv": "metadata.csv",
            "metadata_jsonl": "metadata.jsonl",
            "report": "vector_db_report.md",
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )

    report = build_report(chunks, vectors, args.output, args.dimension)
    (args.output / "vector_db_report.md").write_text(report, encoding="utf-8-sig", newline="\n")

    print(f"vectors={vectors.shape[0]}")
    print(f"dimension={args.dimension}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
