from __future__ import annotations

import argparse
import json
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "work" / "answer_quality" / "answer_prompt_dataset.csv"
DEFAULT_OUTPUT = ROOT / "work" / "answer_quality" / "generated_answers.csv"
DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"


def read_csv_auto(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str).fillna("")
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str).fillna("")


def post_chat_completion(
    *,
    api_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    reasoning_effort: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retry_count: int,
    retry_base_sleep: float,
    retry_max_sleep: float,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    retryable_statuses = {429, 500, 502, 503, 504}
    result: dict[str, object] | None = None

    def wait_before_retry(reason: str, attempt: int) -> None:
        wait_seconds = retry_base_sleep * (2**attempt)
        if retry_max_sleep > 0:
            wait_seconds = min(wait_seconds, retry_max_sleep)
        print(f"{reason}; retrying in {wait_seconds:.1f}s ...")
        time.sleep(wait_seconds)

    for attempt in range(retry_count + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code not in retryable_statuses or attempt >= retry_count:
                raise RuntimeError(f"API HTTP {exc.code}: {body}") from exc
            wait_before_retry(f"API HTTP {exc.code}", attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, socket.timeout) as exc:
            if attempt >= retry_count:
                raise RuntimeError(f"API network error: {exc}") from exc
            wait_before_retry(f"API network error: {exc}", attempt)

    if result is None:
        raise RuntimeError("API request failed without a response.")
    choices = result.get("choices", [])
    if not choices:
        raise RuntimeError(f"API response has no choices: {result}")
    message = choices[0].get("message", {})
    answer = str(message.get("content", "")).strip()
    if not answer:
        raise RuntimeError(f"API response has empty answer: {result}")
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RAG answers from answer_prompt_dataset.csv.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-url", default=os.environ.get("OPENAI_API_URL", DEFAULT_API_URL))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default=os.environ.get("RAG_ANSWER_MODEL", ""))
    parser.add_argument("--limit", type=int, default=0, help="Generate only the first N unanswered rows. 0 means all.")
    parser.add_argument("--sleep", type=float, default=0.3, help="Seconds to sleep between API calls.")
    parser.add_argument("--reasoning-effort", default="", help="Optional OpenAI-compatible reasoning_effort value.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retry-count", type=int, default=3)
    parser.add_argument("--retry-base-sleep", type=float, default=5.0)
    parser.add_argument("--retry-max-sleep", type=float, default=0.0, help="Maximum retry wait seconds. 0 means no cap.")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"Missing API key. Set environment variable {args.api_key_env}.")
    if not args.model:
        raise SystemExit("Missing model. Pass --model or set RAG_ANSWER_MODEL.")

    source = read_csv_auto(args.input)
    if "generated_answer" not in source.columns:
        source["generated_answer"] = ""

    if args.output.exists():
        previous = read_csv_auto(args.output)
        if "question_id" in previous.columns and "generated_answer" in previous.columns:
            previous_answers = dict(zip(previous["question_id"], previous["generated_answer"]))
            source["generated_answer"] = source.apply(
                lambda row: row["generated_answer"] or previous_answers.get(row.get("question_id", ""), ""),
                axis=1,
            )

    generated = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for index, row in source.iterrows():
        if str(row.get("generated_answer", "")).strip():
            continue
        if args.limit and generated >= args.limit:
            break

        question_id = row.get("question_id", f"row_{index + 1}")
        print(f"[{generated + 1}] generating {question_id} ...")
        answer = post_chat_completion(
            api_url=args.api_url,
            api_key=api_key,
            model=args.model,
            system_prompt=str(row.get("system_prompt", "")),
            user_prompt=str(row.get("user_prompt", "")),
            reasoning_effort=args.reasoning_effort,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retry_count=args.retry_count,
            retry_base_sleep=args.retry_base_sleep,
            retry_max_sleep=args.retry_max_sleep,
        )
        source.at[index, "generated_answer"] = answer
        generated += 1
        source.to_csv(args.output, index=False, encoding="utf-8-sig")
        time.sleep(args.sleep)

    source.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"generated={generated}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
