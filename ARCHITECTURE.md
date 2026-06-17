# 프로젝트 진행 과정

## Week 11: 데이터 수집과 앱 초안

- 학과 홈페이지 데이터 수집 범위 결정
- 교육과정, 학과 소개, 진로 및 자격증, 입학 정보 수집
- Flutter 기반 챗봇 UI 초안 구현

## Week 12: RAG 데이터셋과 Vector DB

- 2021~2025 교육과정 자료 정제
- 최종 문서 438개, 청크 445개 구성
- Hash TF-IDF 기반 4,096차원 Vector DB 구축
- 188개 QA 평가셋 정리

## Week 13: 검색 및 rerank 개선

- 질문 의도 분류 규칙 구현
- 연도, 카테고리, 과목명 기반 rerank 적용
- 기대 출처 기준 검색 평가 수행
- 최종 평가셋에서 source hit@1/3/5 검증

## Week 14: 답변 생성과 비교 실험

- 검색 근거를 포함한 Gemini 답변 생성
- 체크포인트와 기대 출처 기반 답변 품질 평가
- 동일한 188개 질문으로 RAG와 No-RAG 비교
- Flutter 앱과 RAG API 서버 연동

## 최종 구현

- `GET /health`: 서버, 문서 수, Gemini 준비 상태 확인
- `POST /ask`: 질문 검색, 답변 생성, 출처 반환
- Gemini API 실패 시 Local fallback 제공
- Flutter 앱에서 답변과 출처 확인 지원
