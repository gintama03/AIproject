# -*- coding: utf-8 -*-
"""Rule-based query intent classifier for Week13 RAG reranking.

This module only classifies user intent. Score boosting/demotion is handled in
the next step so the rules can be reviewed independently.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class IntentRule:
    intent: str
    label: str
    description: str
    keywords: tuple[str, ...]
    preferred_categories: tuple[str, ...]
    demoted_categories: tuple[str, ...]
    related_cases: tuple[str, ...]


INTENT_RULES: tuple[IntentRule, ...] = (
    IntentRule(
        intent="graduation_project",
        label="졸업작품/졸업논문 면제",
        description="졸업작품 제출, 보고서, 심사, 졸업논문 면제 기준을 묻는 질문",
        keywords=("졸업작품", "졸업 작품", "졸업논문", "보고서", "심사", "면제"),
        preferred_categories=("졸업요건",),
        demoted_categories=("교육과정-과목표", "교육과정-교과목개요", "진로및자격증"),
        related_cases=("qa_163",),
    ),
    IntentRule(
        intent="graduation_credit",
        label="졸업 최소 이수학점",
        description="졸업 최소학점, 전공 학점, 단일전공/복수전공 기준을 묻는 질문",
        keywords=("졸업", "학점", "이수학점", "최소 학점", "최소 이수", "단일전공", "복수전공", "전공 학점"),
        preferred_categories=("졸업요건",),
        demoted_categories=("교육과정-교과목개요", "진로및자격증", "입학Q&A"),
        related_cases=("qa_129", "qa_159"),
    ),
    IntentRule(
        intent="general_education",
        label="교양/KY Vision 이수 기준",
        description="교양필수, 중점교양, KY Vision, ESG, 기업가정신 등 교양 이수 기준 질문",
        keywords=("교양", "교양필수", "중점교양", "교양 학점", "ky vision", "기업가정신", "나의 꿈", "미래 디자인", "esg", "dyt"),
        preferred_categories=("교양과정", "졸업요건"),
        demoted_categories=("교육과정-교과목개요", "진로및자격증"),
        related_cases=("qa_062", "qa_119", "qa_135", "qa_179"),
    ),
    IntentRule(
        intent="exact_course_schedule",
        label="특정 과목 학년/학기",
        description="특정 과목이 몇 학년 몇 학기에 배치되는지 묻는 질문",
        keywords=("몇 학년", "몇학년", "몇 학기", "몇학기", "배치", "코스", "교육과정 상", "수강", "들을 수"),
        preferred_categories=("교육과정-과목표",),
        demoted_categories=("졸업요건", "진로및자격증", "입학Q&A"),
        related_cases=("qa_054", "qa_166"),
    ),
    IntentRule(
        intent="curriculum_roadmap",
        label="학년별 교육과정 흐름/로드맵",
        description="학년별 전공 흐름, 커리큘럼 단계, 고학년 프로젝트 과목 흐름 질문",
        keywords=("학년별", "흐름", "로드맵", "커리큘럼", "고학년", "프로젝트 수업", "수업 명칭", "캡스톤", "ai 프로젝트"),
        preferred_categories=("교육과정-로드맵", "교육과정-과목표"),
        demoted_categories=("졸업요건", "교육과정-교과목개요", "진로및자격증"),
        related_cases=("qa_006", "qa_015"),
    ),
    IntentRule(
        intent="course_description",
        label="교과목 개요/수업 내용",
        description="특정 과목에서 무엇을 배우는지, 어떤 기술적 배경을 얻는지 묻는 질문",
        keywords=("무엇을 다루", "어떤 내용", "배우", "기술적 배경", "수업 내용", "과목 개요", "강의"),
        preferred_categories=("교육과정-교과목개요", "교육과정-과목표"),
        demoted_categories=("졸업요건", "입학Q&A", "진로및자격증"),
        related_cases=("qa_145",),
    ),
    IntentRule(
        intent="programming_data_course",
        label="프로그래밍/데이터 도구 과목",
        description="프로그래밍 언어, 코딩 실습, 데이터 분석 도구/라이브러리 관련 질문",
        keywords=("프로그래밍", "코딩", "파이썬", "python", "java", "언어", "라이브러리", "도구", "데이터 분석", "데이터 정제"),
        preferred_categories=("교육과정-과목표", "교육과정-교과목개요"),
        demoted_categories=("졸업요건", "진로및자격증"),
        related_cases=("qa_014", "qa_023", "qa_137"),
    ),
    IntentRule(
        intent="department_intro_history",
        label="학과 소개/설립 배경/명칭 개편",
        description="학과 신설 배경, 교육부 계획, 학과명 변경과 개편 이유 질문",
        keywords=("학과명", "명칭", "개편", "바뀐", "원래", "신설", "정부 부처", "교육부", "첨단분야", "설립", "소개"),
        preferred_categories=("학과소개", "교육목표"),
        demoted_categories=("진로및자격증", "교양과정", "교육과정-과목표"),
        related_cases=("qa_005", "qa_123", "qa_183"),
    ),
    IntentRule(
        intent="admission",
        label="입학/면접/전형",
        description="입학 전 준비, 수시/정시/면접, 전형, 신입생 질문",
        keywords=("입학", "면접", "수시", "정시", "전형", "지원", "신입생", "입학 전"),
        preferred_categories=("입학Q&A", "입학정보", "학과소개"),
        demoted_categories=("졸업요건", "교육과정-교과목개요"),
        related_cases=("qa_020",),
    ),
    IntentRule(
        intent="math_preparation",
        label="수학 부담/입학 전 준비",
        description="수학을 못해도 되는지, 신입생의 수학 부담과 준비 수준을 묻는 질문",
        keywords=("수학", "고등수학", "못하면", "부담", "어렵", "준비"),
        preferred_categories=("입학Q&A", "학과소개", "교육과정-과목표"),
        demoted_categories=("과거연도", "졸업요건"),
        related_cases=("qa_008", "qa_012"),
    ),
    IntentRule(
        intent="career_certificate",
        label="진로/취업/자격증",
        description="졸업 후 진로, 취업 분야, 직무, 자격증, 인증 관련 질문",
        keywords=("진로", "졸업 후", "졸업후", "취업", "자격증", "인증", "직무", "기업", "졸업생", "진출"),
        preferred_categories=("진로및자격증", "학과소개"),
        demoted_categories=("졸업요건", "교육과정-과목표"),
        related_cases=("qa_016",),
    ),
    IntentRule(
        intent="practice_environment",
        label="실습환경/GPU/장비",
        description="AI 모델 학습 환경, GPU, 서버, PC, 실습 장비 관련 질문",
        keywords=("gpu", "실습 환경", "실습환경", "고성능", "컴퓨터", "pc", "서버", "장비", "모델 학습"),
        preferred_categories=("학과소개", "입학정보"),
        demoted_categories=("진로및자격증", "졸업요건", "교육과정-교과목개요"),
        related_cases=("qa_019",),
    ),
    IntentRule(
        intent="academic_policy",
        label="학사 규정/출석/학점 인정",
        description="조기취업, 출석, 학점 인정, 구제 방법처럼 학사 운영 기준을 묻는 질문",
        keywords=("조기취업", "출석", "학점 인정", "인정받", "구제", "지침", "규정", "모자라"),
        preferred_categories=("졸업요건", "학사일정", "입학정보"),
        demoted_categories=("진로및자격증", "교육과정-교과목개요"),
        related_cases=("qa_096",),
    ),
)


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", "", str(query or "").lower())


def extract_years(query: str) -> list[str]:
    return sorted(set(re.findall(r"20[0-9]{2}", str(query or ""))))


def matched_keywords(query: str, keywords: tuple[str, ...]) -> list[str]:
    normalized = normalize_query(query)
    matched: list[str] = []
    for keyword in keywords:
        if normalize_query(keyword) in normalized:
            matched.append(keyword)
    return matched


def classify_query_intent(query: str) -> dict[str, object]:
    """Classify a query into the best matching RAG intent rule."""
    candidates: list[tuple[int, int, IntentRule, list[str]]] = []
    for order, rule in enumerate(INTENT_RULES):
        hits = matched_keywords(query, rule.keywords)
        if hits:
            candidates.append((len(hits), -order, rule, hits))

    if not candidates:
        return {
            "intent": "general",
            "label": "일반 질문",
            "description": "명확한 의도가 감지되지 않은 일반 학과 질문",
            "matched_keywords": [],
            "years": extract_years(query),
            "preferred_categories": [],
            "demoted_categories": [],
            "related_cases": [],
        }

    _, _, rule, hits = max(candidates, key=lambda item: (item[0], item[1]))
    result = asdict(rule)
    result["matched_keywords"] = hits
    result["years"] = extract_years(query)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a RAG query intent.")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    print(json.dumps(classify_query_intent(args.query), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
