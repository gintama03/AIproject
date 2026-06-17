# KY AI RAG Chatbot

건양대학교 인공지능학과의 공식 자료를 기반으로 교육과정, 졸업요건, 학과 소개, 진로 및 입학 정보를 안내하는 RAG 챗봇입니다. Flutter 앱이 로컬 Python RAG API에 질문을 보내고, 서버는 Vector DB에서 근거 문서를 검색한 뒤 Gemini API 또는 로컬 fallback 방식으로 답변과 출처를 반환합니다.

## 핵심 결과

| 항목 | 결과 |
|---|---:|
| 최종 RAG 문서 | 438개 |
| 최종 RAG 청크 | 445개 |
| 평가 질문 | 188개 |
| RAG PASS | 188개 |
| No-RAG PASS | 23개 |
| RAG PASS rate | 100.00% |
| No-RAG PASS rate | 12.23% |

> PASS는 본 프로젝트에서 정의한 체크포인트와 기대 출처 기준을 충족했다는 의미이며, 모든 실제 질문에 대한 절대적 정확도 100%를 의미하지 않습니다.

## 시스템 구조

```text
사용자
  -> Flutter 앱
  -> POST /ask
  -> Python RAG API 서버
  -> Vector DB 검색 및 rerank
  -> Gemini API 답변 또는 Local fallback
  -> 답변과 출처 JSON 반환
  -> Flutter 앱 표시
```

- 프론트엔드: Flutter
- 백엔드: Python 표준 HTTP 서버
- 벡터화: `local_hash_tfidf_char_word_v1`
- 벡터 차원: 4,096
- 유사도: Cosine similarity
- 검색 보정: 질문 의도, 연도, 카테고리, 과목명 기반 rule-based rerank
- 답변 생성: Gemini 2.5 Flash Lite 기본값
- 평가: 188개 QA 세트, source hit, checkpoint ratio, PASS/REVIEW/FAIL

자세한 흐름은 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)를 참고하세요.

## 저장소 구조

```text
.
├─ lib/                         Flutter 앱 코드
├─ assets/                      앱 이미지
├─ android/, web/, windows/     Flutter 플랫폼 코드
├─ tools/                       RAG, 검색, 평가 Python 코드
├─ scripts/                     실행 스크립트
├─ data/
│  ├─ final_rag/                최종 문서 및 청크
│  ├─ vector_db/                로컬 Vector DB
│  └─ evaluation/               188개 QA 평가셋
├─ results/
│  ├─ retrieval/                검색 평가 결과
│  ├─ answer_quality/            RAG/No-RAG 답변 평가
│  └─ rag_vs_no_rag/             최종 비교 결과
└─ docs/                        구조 및 개발 과정 문서
```

## 환경 설정

### 1. Python 환경

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Gemini 답변 생성을 사용할 경우 환경변수를 설정합니다.

```powershell
$env:GEMINI_API_KEY="your_api_key"
```

API 키가 없거나 Gemini 호출에 실패하면 서버는 검색된 로컬 문서를 기반으로 fallback 응답을 반환합니다. 실제 키는 저장소에 커밋하지 않습니다.

### 2. Flutter 환경

```powershell
flutter pub get
```

## 실행 방법

### 1. RAG API 서버 실행

저장소 루트에서 실행합니다.

```powershell
.\scripts\run_rag_api_server.ps1
```

기본 주소는 `http://127.0.0.1:8000`입니다.

### 2. Flutter 앱 실행

```powershell
flutter run -d chrome
```

Android 실기기에서 USB 디버깅으로 실행할 경우 PC의 로컬 포트를 연결합니다.

```powershell
adb reverse tcp:8000 tcp:8000
flutter run
```

## API

### `GET /health`

RAG API 서버 상태, 로드된 문서 수, Gemini API 준비 상태를 확인합니다.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

### `POST /ask`

질문을 서버에 전달하고 답변과 출처를 반환받습니다.

```powershell
$body = @{
  question = "2025년 인공지능학과 졸업 최소 학점은?"
  top_k = 3
  use_llm = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/ask `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

응답에는 `answer`, `sources`, `mode`, `error`가 포함됩니다. 출처의 `score`는 정답 확률이 아니라 Cosine 유사도와 rerank 보정을 반영한 검색 관련도 점수입니다.

## 데이터 및 검색 재현

최종 청크에서 Vector DB를 다시 생성합니다.

```powershell
python tools/build_local_vector_db.py
```

질문으로 로컬 검색을 확인합니다.

```powershell
python tools/search_local_vector_db.py `
  --query "2025년 졸업 최소 학점은?" `
  --top-k 3
```

188개 QA 세트로 검색 성능을 다시 평가합니다.

```powershell
python tools/evaluate_retrieval.py
```

실행 결과는 Git에서 제외되는 `work/` 아래에 생성됩니다. 제출 당시의 최종 결과는 `results/`에서 확인할 수 있습니다.

## RAG vs No-RAG

동일한 188개 질문을 사용해 다음 두 조건을 비교했습니다.

- No-RAG: 학과 근거 문서 없이 Gemini에 질문 전달
- RAG: Vector DB에서 검색한 학과 자료와 질문을 Gemini에 함께 전달

| 평가 항목 | No-RAG | RAG |
|---|---:|---:|
| 평가 질문 수 | 188 | 188 |
| PASS | 23 | 188 |
| REVIEW | 68 | 0 |
| FAIL | 97 | 0 |
| PASS rate | 12.23% | 100.00% |
| 평균 체크포인트 충족률 | 45.07% | 100.00% |
| 평균 출처 충족률 | 0.00% | 100.00% |

비교 결과 원본은 [results/rag_vs_no_rag](results/rag_vs_no_rag)에 있습니다.

## 한계

- Hash TF-IDF 방식은 딥러닝 의미 임베딩보다 문맥 이해가 제한적입니다.
- 포괄적인 질문에서도 상위 청크 수가 적으면 일부 항목만 답변할 수 있습니다.
- Gemini API는 결제, 쿼터, 네트워크 상태의 영향을 받습니다.
- Local fallback은 별도의 로컬 생성형 모델이 아니라 검색 문서를 중심으로 반환하는 대체 응답입니다.
- 공지사항처럼 자주 변경되는 자료는 자동 갱신되지 않습니다.

## 팀원 및 역할

| 팀원 | 실제 수행 역할 |
|---|---|
| 강준형 | 프로젝트 총괄, Flutter 앱 구현, RAG API 연동, 최종 시연 환경 정리 |
| 박희재 | RAG 검색·답변 파이프라인, QA 평가셋, RAG vs No-RAG 비교 및 결과 분석 |
| 이세빈 | 학과 데이터 수집·정제, 교육과정 PDF 처리, RAG 데이터셋 및 Vector DB 구축 |

## 데이터 출처

- 건양대학교 인공지능학과 공식 홈페이지
- 건양대학교 인공지능학과 2021~2025 교육과정 자료
- Google Gemini API

원본 수업 가이드, 제출용 보고서, PPT/PDF 발표자료와 API 키는 저장소에서 제외했습니다.
