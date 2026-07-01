#!/usr/bin/env python3
"""Redrob candidate ranker.

This script produces a valid top-100 submission from candidates.jsonl.

Design goals:
- CPU-only, no network, no model downloads.
- Stream the large JSONL file; keep only compact numeric features in memory.
- Prefer high-signal production retrieval/ranking candidates over AI keyword stuffers.
- Apply explicit penalties for honeypot-like inconsistencies.
- Train a deterministic supervised model on pseudo labels, validate ranking
  quality on a stratified holdout, then generate grounded reasoning from facts.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import importlib.util
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np


EVAL_DATE = dt.date(2026, 6, 29)

FEATURE_NAMES = [
    "archetype",
    "semantic_match",
    "title_match",
    "career_evidence",
    "skill_depth",
    "behavior",
    "logistics",
    "trust",
    "retrieval_depth",
    "production_evidence",
    "eval_framework",
    "vector_db_exp",
    "experience_band_fit",
    "recency",
    "hiring_readiness",
    "career_stability",
    "skill_credibility",
    "role_x_semantic",
    "company_quality",
    "nlp_ir_focus",
]
AUGMENTED_FEATURE_NAMES = FEATURE_NAMES + ["risk", "hard_trap"]

DEFAULT_RISK_PENALTY = 0.0
DEFAULT_HARD_TRAP_PENALTY = 0.0


@dataclass
class RankerModel:
    estimator: Any
    model_name: str
    class_values: np.ndarray
    feature_names: List[str]
    feature_importances: np.ndarray
    risk_penalty: float
    hard_trap_penalty: float
    validation_metrics: Dict[str, Any]

SERVICE_COMPANIES = {
    "tcs",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl",
    "mindtree",
}

NON_TECH_TITLES = {
    "marketing manager",
    "operations manager",
    "sales executive",
    "hr manager",
    "accountant",
    "civil engineer",
    "mechanical engineer",
    "graphic designer",
    "content writer",
    "customer support",
    "project manager",
    "business analyst",
}

PREFERRED_CITY_TERMS = {
    "pune",
    "noida",
    "delhi",
    "gurgaon",
    "gurugram",
    "ncr",
    "mumbai",
    "hyderabad",
    "bangalore",
    "bengaluru",
}

PRIMARY_EVIDENCE_TERMS = [
    ("hybrid retrieval", 9.0),
    ("dense retrieval", 8.5),
    ("learning to rank", 8.5),
    ("ranking pipeline", 8.0),
    ("candidate jd matching", 8.0),
    ("search ranking", 7.5),
    ("recruiter facing search", 7.0),
    ("information retrieval", 7.0),
    ("retrieval", 6.5),
    ("ranking", 6.5),
    ("ranker", 6.0),
    ("bm25", 6.0),
    ("ndcg", 6.0),
    ("mrr", 5.0),
    ("recall k", 4.2),
    ("offline evaluation", 5.5),
    ("online evaluation", 5.0),
    ("a b", 4.7),
    ("ab test", 4.7),
    ("behavioral signal", 4.4),
    ("embedding", 4.8),
    ("embeddings", 4.8),
    ("vector", 4.2),
    ("faiss", 4.0),
    ("pinecone", 3.8),
    ("weaviate", 3.8),
    ("qdrant", 3.8),
    ("milvus", 3.6),
    ("opensearch", 3.4),
    ("elasticsearch", 3.4),
    ("recommendation", 4.2),
    ("recommender", 4.2),
    ("rag", 3.5),
    ("llm based re ranker", 3.2),
    ("fine tuning", 2.9),
    ("lora", 2.2),
    ("qlora", 2.2),
    ("python", 3.2),
    ("production ml", 4.5),
    ("deployed", 3.2),
    ("serving", 3.0),
    ("latency", 2.8),
    ("queries per month", 3.4),
]

SKILL_WEIGHTS = {
    "learning to rank": 1.00,
    "information retrieval": 0.96,
    "bm25": 0.90,
    "embeddings": 0.88,
    "embedding": 0.88,
    "rag": 0.82,
    "recommendation systems": 0.82,
    "faiss": 0.78,
    "pinecone": 0.76,
    "weaviate": 0.76,
    "qdrant": 0.76,
    "milvus": 0.74,
    "opensearch": 0.70,
    "elasticsearch": 0.70,
    "nlp": 0.66,
    "llm": 0.62,
    "fine tuning llms": 0.58,
    "fine tuning": 0.56,
    "qlora": 0.50,
    "lora": 0.46,
    "python": 0.62,
    "pytorch": 0.56,
    "scikit learn": 0.52,
    "tensorflow": 0.46,
    "mlops": 0.46,
    "bentoml": 0.36,
    "kubernetes": 0.30,
}

REASON_TERMS = [
    ("hybrid retrieval", "hybrid retrieval"),
    ("dense retrieval", "dense retrieval"),
    ("learning to rank", "learning-to-rank"),
    ("ranking pipeline", "ranking pipelines"),
    ("search ranking", "search/ranking"),
    ("information retrieval", "information retrieval"),
    ("retrieval", "retrieval"),
    ("ranking", "ranking"),
    ("bm25", "BM25"),
    ("ndcg", "NDCG"),
    ("mrr", "MRR"),
    ("offline evaluation", "offline evaluation"),
    ("online evaluation", "online evaluation"),
    ("a b", "A/B testing"),
    ("embedding", "embeddings"),
    ("vector", "vector search"),
    ("faiss", "FAISS"),
    ("pinecone", "Pinecone"),
    ("weaviate", "Weaviate"),
    ("qdrant", "Qdrant"),
    ("milvus", "Milvus"),
    ("opensearch", "OpenSearch"),
    ("elasticsearch", "Elasticsearch"),
    ("recommendation", "recommendation systems"),
    ("rag", "RAG"),
    ("fine tuning", "fine-tuning"),
    ("python", "Python"),
    ("production ml", "production ML"),
]


def open_candidates(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def norm_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9+#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_texts(candidate: dict) -> Tuple[str, str, str, str]:
    profile = candidate["profile"]
    profile_parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_company", ""),
        profile.get("current_industry", ""),
        profile.get("location", ""),
        profile.get("country", ""),
    ]

    career_parts: List[str] = []
    for job in candidate.get("career_history", []):
        career_parts.extend(
            [
                job.get("title", ""),
                job.get("company", ""),
                job.get("industry", ""),
                job.get("description", ""),
            ]
        )

    skill_parts = [skill.get("name", "") for skill in candidate.get("skills", [])]
    education_parts: List[str] = []
    for edu in candidate.get("education", []):
        education_parts.extend(
            [
                edu.get("institution", ""),
                edu.get("degree", ""),
                edu.get("field_of_study", ""),
                edu.get("tier", ""),
            ]
        )

    profile_text = norm_text(" ".join(str(x) for x in profile_parts if x))
    career_text = norm_text(" ".join(str(x) for x in career_parts if x))
    skill_text = norm_text(" ".join(str(x) for x in skill_parts if x))
    all_text = " ".join(part for part in [profile_text, career_text, skill_text, norm_text(" ".join(education_parts))] if part)
    return profile_text, career_text, skill_text, all_text


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def numeric_value(mapping: dict, key: str, default: float) -> float:
    value = mapping.get(key, default)
    if value is None:
        return default
    return float(value)


def notice_days(signals: dict) -> int:
    return int(numeric_value(signals, "notice_period_days", 180.0))


def parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def months_between(start: str | None, end: str | None) -> int | None:
    start_date = parse_date(start)
    end_date = parse_date(end) or EVAL_DATE
    if start_date is None:
        return None
    return max(0, (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month))


def contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def archetype_score(profile_text: str) -> Tuple[float, str]:
    if "senior ai engineer with" in profile_text:
        return 1.00, "senior_ai"
    if "machine learning engineer with" in profile_text:
        return 0.82, "ml_engineer"
    if "data scientist ml engineer with" in profile_text:
        return 0.50, "ds_ml"
    if "software data professional" in profile_text:
        return 0.28, "software_data"
    if "software engineer with" in profile_text:
        return 0.17, "software_engineer"
    if contains_any(profile_text, ["generative ai explorer", "ai enthusiast", "building with llms"]):
        return 0.04, "ai_enthusiast"
    if "professional with" in profile_text:
        return 0.02, "generic_professional"
    return 0.10, "other"


def semantic_score(all_text: str) -> float:
    total = 0.0
    for phrase, weight in PRIMARY_EVIDENCE_TERMS:
        if phrase in all_text:
            total += weight

    # Reward combinations the JD explicitly cares about.
    if "retrieval" in all_text and "ranking" in all_text:
        total += 6.0
    if "embedding" in all_text and contains_any(all_text, ["faiss", "pinecone", "weaviate", "qdrant", "milvus"]):
        total += 4.0
    if contains_any(all_text, ["ndcg", "mrr", "map"]) and contains_any(all_text, ["offline evaluation", "online evaluation", "a b"]):
        total += 4.0
    if "candidate jd matching" in all_text or "candidate job matching" in all_text:
        total += 5.0

    return clamp(total / 85.0)


def title_score(title_text: str) -> float:
    senior = contains_any(title_text, ["senior", "lead", "staff", "principal"])
    if contains_any(title_text, ["search engineer", "recommendation systems engineer"]):
        return 0.90 if senior else 0.84
    if contains_any(title_text, ["machine learning engineer", "ai engineer", "nlp engineer", "applied ml engineer"]):
        return 0.95 if senior else 0.78
    if "applied scientist" in title_text:
        return 0.88 if senior else 0.72
    if "senior data scientist" in title_text:
        return 0.72
    if contains_any(title_text, ["data scientist", "ml engineer", "ai specialist"]):
        return 0.55
    if contains_any(title_text, ["backend engineer", "data engineer", "analytics engineer", "software engineer"]):
        return 0.30 if senior else 0.22
    if title_text in NON_TECH_TITLES:
        return 0.0
    return 0.08


def skill_depth_score(candidate: dict) -> Tuple[float, int, int]:
    prof_factor = {"beginner": 0.25, "intermediate": 0.55, "advanced": 0.82, "expert": 1.00}
    total = 0.0
    expert_near_zero = 0
    expert_under_year = 0

    for skill in candidate.get("skills", []):
        name = norm_text(skill.get("name", ""))
        proficiency = skill.get("proficiency", "beginner")
        duration = numeric_value(skill, "duration_months", 0.0)
        endorsements = numeric_value(skill, "endorsements", 0.0)

        if proficiency == "expert" and duration <= 3:
            expert_near_zero += 1
        if proficiency == "expert" and duration < 12:
            expert_under_year += 1

        base = 0.0
        for term, weight in SKILL_WEIGHTS.items():
            if term in name:
                base = max(base, weight)
        if base <= 0.0:
            continue

        duration_factor = 0.35 + 0.65 * clamp(duration / 60.0)
        endorse_factor = 0.75 + 0.25 * clamp(math.log1p(endorsements) / math.log1p(60))
        total += base * prof_factor.get(proficiency, 0.30) * duration_factor * endorse_factor

    return clamp(total / 8.0), expert_near_zero, expert_under_year


def services_only(candidate: dict) -> bool:
    jobs = candidate.get("career_history", [])
    if not jobs:
        return False
    service_like = 0
    for job in jobs:
        company = norm_text(job.get("company", ""))
        industry = norm_text(job.get("industry", ""))
        if industry in {"it services", "consulting"} or any(name in company for name in SERVICE_COMPANIES):
            service_like += 1
    return service_like == len(jobs)


def career_evidence_score(candidate: dict, career_text: str) -> Tuple[float, bool]:
    product_industries = {
        "software",
        "internet",
        "fintech",
        "e commerce",
        "food delivery",
        "saas",
        "ai ml",
        "edtech",
        "marketplace",
        "media",
        "insurance tech",
        "conversational ai",
        "consumer electronics",
        "transportation",
        "adtech",
    }
    jobs = candidate.get("career_history", [])
    relevant_months = 0
    product_jobs = 0
    shipping_hits = 0.0

    for job in jobs:
        job_text = norm_text(" ".join([job.get("title", ""), job.get("industry", ""), job.get("description", "")]))
        if contains_any(job_text, ["retrieval", "ranking", "search", "recommendation", "embedding", "candidate jd", "ndcg"]):
            relevant_months += int(job.get("duration_months", 0) or 0)
        if norm_text(job.get("industry", "")) in product_industries:
            product_jobs += 1
        if contains_any(job_text, ["built", "owned", "designed", "shipped", "deployed", "serving", "latency", "queries"]):
            shipping_hits += 1.0

    relevant_score = clamp(relevant_months / 60.0)
    product_score = clamp(product_jobs / max(1, len(jobs)))
    shipping_score = clamp(shipping_hits / max(1, len(jobs)))
    eval_score = 1.0 if contains_any(career_text, ["ndcg", "mrr", "map", "offline evaluation", "online evaluation", "a b"]) else 0.0
    service_only = services_only(candidate)

    score = 0.44 * relevant_score + 0.22 * product_score + 0.22 * shipping_score + 0.12 * eval_score
    if service_only:
        score *= 0.72
    return clamp(score), service_only


def behavior_score(signals: dict) -> float:
    last_active = parse_date(signals.get("last_active_date"))
    if last_active is None:
        recency = 0.0
    else:
        days = max(0, (EVAL_DATE - last_active).days)
        if days <= 14:
            recency = 1.00
        elif days <= 30:
            recency = 0.88
        elif days <= 60:
            recency = 0.70
        elif days <= 120:
            recency = 0.42
        else:
            recency = 0.15

    response_rate = clamp(numeric_value(signals, "recruiter_response_rate", 0.0))
    response_hours = numeric_value(signals, "avg_response_time_hours", 999.0)
    response_time = 1.0 / (1.0 + response_hours / 48.0)
    profile_complete = clamp(numeric_value(signals, "profile_completeness_score", 0.0) / 100.0)
    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.0
    views = clamp(math.log1p(numeric_value(signals, "profile_views_received_30d", 0.0)) / math.log1p(180))
    saved = clamp(math.log1p(numeric_value(signals, "saved_by_recruiters_30d", 0.0)) / math.log1p(25))
    search = clamp(math.log1p(numeric_value(signals, "search_appearance_30d", 0.0)) / math.log1p(1000))
    interview = clamp(numeric_value(signals, "interview_completion_rate", 0.0))
    acceptance_raw = numeric_value(signals, "offer_acceptance_rate", -1.0)
    acceptance = 0.50 if acceptance_raw < 0 else clamp(acceptance_raw)
    github_raw = numeric_value(signals, "github_activity_score", -1.0)
    github = 0.25 if github_raw < 0 else clamp(github_raw / 100.0)

    return clamp(
        0.17 * recency
        + 0.15 * open_to_work
        + 0.15 * response_rate
        + 0.07 * response_time
        + 0.08 * profile_complete
        + 0.07 * views
        + 0.08 * saved
        + 0.06 * search
        + 0.08 * interview
        + 0.04 * acceptance
        + 0.05 * github
    )


def logistics_score(profile: dict, signals: dict) -> float:
    years = float(profile.get("years_of_experience", 0.0) or 0.0)
    if 5.0 <= years <= 9.0:
        exp_score = 1.0 - 0.12 * abs(years - 7.0) / 2.0
    elif 4.0 <= years < 5.0 or 9.0 < years <= 11.0:
        exp_score = 0.62
    elif 3.0 <= years < 4.0 or 11.0 < years <= 13.0:
        exp_score = 0.35
    else:
        exp_score = 0.16
    exp_score = clamp(exp_score)

    country = norm_text(profile.get("country", ""))
    location = norm_text(profile.get("location", ""))
    willing = bool(signals.get("willing_to_relocate"))
    if country == "india" and contains_any(location, ["pune", "noida"]):
        location_score = 1.00
    elif country == "india" and contains_any(location, PREFERRED_CITY_TERMS):
        location_score = 0.86
    elif country == "india":
        location_score = 0.68 if willing else 0.58
    else:
        location_score = 0.45 if willing else 0.14

    notice = notice_days(signals)
    if notice <= 15:
        notice_score = 1.00
    elif notice <= 30:
        notice_score = 0.90
    elif notice <= 45:
        notice_score = 0.75
    elif notice <= 60:
        notice_score = 0.58
    elif notice <= 90:
        notice_score = 0.34
    elif notice <= 120:
        notice_score = 0.18
    else:
        notice_score = 0.05

    work_mode = norm_text(signals.get("preferred_work_mode", ""))
    work_score = {"hybrid": 1.00, "flexible": 0.95, "onsite": 0.82, "remote": 0.74}.get(work_mode, 0.70)

    return clamp(0.42 * exp_score + 0.32 * location_score + 0.20 * notice_score + 0.06 * work_score)


def trust_score(candidate: dict, signals: dict) -> float:
    profile_complete = clamp(numeric_value(signals, "profile_completeness_score", 0.0) / 100.0)
    github_raw = numeric_value(signals, "github_activity_score", -1.0)
    github = 0.35 if github_raw < 0 else clamp(github_raw / 100.0)
    verified = (
        float(bool(signals.get("verified_email")))
        + float(bool(signals.get("verified_phone")))
        + float(bool(signals.get("linkedin_connected")))
    ) / 3.0

    assessments = signals.get("skill_assessment_scores", {}) or {}
    if assessments:
        assessment = clamp(sum(float(v) for v in assessments.values()) / (100.0 * len(assessments)))
    else:
        assessment = 0.45

    tier_score = 0.40
    tier_map = {"tier_1": 1.00, "tier_2": 0.76, "tier_3": 0.52, "tier_4": 0.32, "unknown": 0.40}
    for edu in candidate.get("education", []):
        tier_score = max(tier_score, tier_map.get(edu.get("tier", "unknown"), 0.40))

    return clamp(0.25 * profile_complete + 0.25 * github + 0.20 * verified + 0.18 * assessment + 0.12 * tier_score)


def retrieval_depth_score(career_text: str) -> float:
    phrases = ["hybrid retrieval", "dense retrieval", "learning to rank", "ranking pipeline", 
               "search ranking", "information retrieval", "retrieval", "ranking", "bm25", "ndcg", "mrr", "map"]
    count = sum(career_text.count(p) for p in phrases)
    return clamp(count / 10.0)


def production_evidence_score(career_text: str) -> float:
    words = ["built", "shipped", "deployed", "serving", "latency", "production", "scale", "queries", "throughput"]
    count = sum(career_text.count(w) for w in words)
    return clamp(count / 8.0)


def eval_framework_score(career_text: str) -> float:
    terms = ["ndcg", "mrr", "map", "precision", "recall", "a b", "ab test", "offline evaluation", "online evaluation", "evaluation framework"]
    count = sum(career_text.count(t) for t in terms)
    return clamp(count / 4.0)


def vector_db_exp_score(all_text: str) -> float:
    dbs = ["pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch", "elasticsearch"]
    count = sum(all_text.count(db) for db in dbs)
    return clamp(count / 3.0)


def experience_band_fit_score(years: float) -> float:
    if 5.0 <= years <= 9.0:
        return 1.0 - 0.12 * abs(years - 7.0) / 2.0
    elif 4.0 <= years < 5.0 or 9.0 < years <= 11.0:
        return 0.62
    elif 3.0 <= years < 4.0 or 11.0 < years <= 13.0:
        return 0.35
    else:
        return 0.15


def hiring_readiness_score(signals: dict) -> float:
    otw = 1.0 if signals.get("open_to_work_flag") else 0.0
    notice = notice_days(signals)
    if notice <= 15:
        notice_factor = 1.0
    elif notice <= 30:
        notice_factor = 0.9
    elif notice <= 60:
        notice_factor = 0.7
    elif notice <= 90:
        notice_factor = 0.4
    else:
        notice_factor = 0.1
    resp = clamp(numeric_value(signals, "recruiter_response_rate", 0.0))
    return clamp(0.4 * otw + 0.3 * notice_factor + 0.3 * resp)


def career_stability_score(candidate: dict) -> float:
    jobs = candidate.get("career_history", [])
    if not jobs:
        return 0.5
    short_stints = 0
    total_months = 0
    for job in jobs:
        months = int(job.get("duration_months", 0) or 0)
        total_months += months
        if months < 18:
            short_stints += 1
    avg_tenure = total_months / max(1, len(jobs))
    stability = 1.0 - (short_stints / len(jobs)) * 0.5
    if avg_tenure < 18:
        stability *= 0.7
    return clamp(stability)


def skill_credibility_score(expert_near_zero: int, expert_under_year: int) -> float:
    penalty = 0.0
    if expert_near_zero > 0:
        penalty += 0.25 * expert_near_zero
    if expert_under_year > 0:
        penalty += 0.10 * expert_under_year
    return clamp(1.0 - penalty)


def company_quality_score(candidate: dict) -> float:
    jobs = candidate.get("career_history", [])
    if not jobs:
        return 0.5
    product_industries = {
        "software", "internet", "fintech", "e commerce", "food delivery", "saas",
        "ai ml", "edtech", "marketplace", "media", "insurance tech", "conversational ai",
        "consumer electronics", "transportation", "adtech"
    }
    product_jobs = 0
    consulting_jobs = 0
    for job in jobs:
        company = norm_text(job.get("company", ""))
        industry = norm_text(job.get("industry", ""))
        if industry in {"it services", "consulting"} or any(name in company for name in SERVICE_COMPANIES):
            consulting_jobs += 1
        elif industry in product_industries:
            product_jobs += 1
    
    total = len(jobs)
    score = (product_jobs * 1.0 + (total - product_jobs - consulting_jobs) * 0.5) / total
    if consulting_jobs == total:
        score = 0.2
    return clamp(score)


def nlp_ir_focus_score(candidate: dict) -> float:
    nlp_ir_terms = {"nlp", "information retrieval", "learning to rank", "retrieval", "ranking", "search", 
                    "embeddings", "embedding", "rag", "recommendation", "vector search", "llm"}
    cv_speech_terms = {"computer vision", "image classification", "speech recognition", "robotics", 
                       "object detection", "image segmentation", "tts", "gans"}
    
    nlp_ir_count = 0
    cv_speech_count = 0
    for skill in candidate.get("skills", []):
        name = norm_text(skill.get("name", ""))
        if any(term in name for term in nlp_ir_terms):
            nlp_ir_count += 1
        if any(term in name for term in cv_speech_terms):
            cv_speech_count += 1
            
    if nlp_ir_count == 0 and cv_speech_count > 0:
        return 0.2
    elif nlp_ir_count > 0 and cv_speech_count > 0:
        return 0.7
    elif nlp_ir_count > 0:
        return 1.0
    return 0.5


def extract_year_mentions(text: str) -> List[float]:
    values: List[float] = []
    # norm_text turns "7.2 years" into "7 2 years"; parse that as 7.2
    # instead of accidentally matching the trailing "2 years".
    pattern = r"\b([0-9]{1,2})(?:\s+([0-9]))?\+?\s+years?\s+of(?:\s+hands\s+on)?\s+experience"
    for match in re.finditer(pattern, text):
        try:
            whole = float(match.group(1))
            decimal = match.group(2)
            if decimal is not None:
                whole += float(decimal) / 10.0
            values.append(whole)
        except ValueError:
            pass
    return values


def risk_score(
    candidate: dict,
    profile_text: str,
    career_text: str,
    all_text: str,
    archetype: str,
    semantic: float,
    title: float,
    service_only: bool,
    expert_near_zero: int,
    expert_under_year: int,
) -> Tuple[float, bool]:
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    years = float(profile.get("years_of_experience", 0.0) or 0.0)
    risk = 0.0
    hard_trap = False

    mentions = extract_year_mentions(profile_text)
    if mentions and min(abs(years - value) for value in mentions) > 1.75:
        risk += 0.75
        hard_trap = True

    career_months = sum(int(job.get("duration_months", 0) or 0) for job in candidate.get("career_history", []))
    if years > 4 and abs(career_months / 12.0 - years) > 3.0:
        risk += 0.62
        hard_trap = True

    date_mismatch_count = 0
    for job in candidate.get("career_history", []):
        computed = months_between(job.get("start_date"), job.get("end_date"))
        if computed is not None and abs(computed - int(job.get("duration_months", 0) or 0)) > 3:
            date_mismatch_count += 1
    if date_mismatch_count >= 2:
        risk += 0.50
        hard_trap = True

    if expert_near_zero >= 3 or expert_under_year >= 8:
        risk += 0.80
        hard_trap = True

    current_title = norm_text(profile.get("current_title", ""))
    non_tech_title = current_title in NON_TECH_TITLES
    if non_tech_title and contains_any(all_text, ["llm", "rag", "langchain", "openai", "genai", "chatgpt"]):
        risk += 0.55
    if archetype in {"ai_enthusiast", "generic_professional"}:
        risk += 0.35
    if "marketing manager" in profile_text and current_title != "marketing manager":
        risk += 0.20

    has_ir = contains_any(all_text, ["retrieval", "ranking", "search", "recommendation", "information retrieval", "ndcg"])
    if contains_any(all_text, ["langchain", "openai api", "chatgpt"]) and not has_ir:
        risk += 0.35
    if contains_any(all_text, ["computer vision", "image classification", "speech recognition", "robotics"]) and not has_ir:
        risk += 0.25

    if service_only:
        risk += 0.22 if semantic > 0.55 and title > 0.60 else 0.45

    last_active = parse_date(signals.get("last_active_date"))
    if last_active and (EVAL_DATE - last_active).days > 150:
        risk += 0.18
    if numeric_value(signals, "recruiter_response_rate", 0.0) < 0.10:
        risk += 0.12
    if notice_days(signals) >= 120:
        risk += 0.10

    short_senior_stints = 0
    for job in candidate.get("career_history", []):
        job_title = norm_text(job.get("title", ""))
        if contains_any(job_title, ["senior", "staff", "lead", "principal"]) and int(job.get("duration_months", 0) or 0) < 18:
            short_senior_stints += 1
    if short_senior_stints >= 2:
        risk += 0.10

    return clamp(risk), hard_trap


def pseudo_relevance(
    archetype: str,
    semantic: float,
    title: float,
    skill: float,
    career: float,
    behavior: float,
    logistics: float,
    trust: float,
    retrieval_depth: float,
    production_evidence: float,
    eval_framework: float,
    vector_db_exp: float,
    experience_band_fit: float,
    hiring_readiness: float,
    career_stability: float,
    skill_credibility: float,
    company_quality: float,
    nlp_ir_focus: float,
    risk: float,
    hard_trap: bool,
) -> float:
    if hard_trap or risk >= 0.85:
        return 0.0

    # 1. Base Role Alignment (Archetype + Title + Semantic)
    arch_score = {
        "senior_ai": 1.0,
        "ml_engineer": 0.82,
        "ds_ml": 0.50,
        "software_data": 0.28,
        "software_engineer": 0.17,
        "ai_enthusiast": 0.04,
        "generic_professional": 0.02,
        "other": 0.10
    }.get(archetype, 0.10)
    
    role_fit = 0.45 * arch_score + 0.30 * title + 0.25 * semantic
    
    # 2. Technical and Experience Quality
    tech_depth = (
        0.30 * retrieval_depth +
        0.20 * production_evidence +
        0.20 * eval_framework +
        0.15 * vector_db_exp +
        0.15 * skill
    )
    
    base_score = (
        1.6 * role_fit +
        0.8 * tech_depth +
        0.4 * experience_band_fit +
        0.4 * nlp_ir_focus +
        0.4 * company_quality +
        0.4 * career
    )
    
    # 3. Adjust by logistics, behavioral readiness and credibility
    logistics_factor = 0.6 + 0.4 * logistics
    behavior_factor = 0.7 + 0.3 * behavior
    readiness_factor = 0.8 + 0.2 * hiring_readiness
    stability_factor = 0.8 + 0.2 * career_stability
    credibility_factor = 0.8 + 0.2 * skill_credibility
    trust_factor = 0.9 + 0.1 * trust
    
    final_rel = base_score * logistics_factor * behavior_factor * readiness_factor * stability_factor * credibility_factor * trust_factor
    
    if risk > 0.30:
        final_rel *= (1.0 - (risk - 0.30))
        
    return float(clamp(final_rel, 0.0, 4.0))




def extract_features(candidate: dict) -> Tuple[List[float], float, bool, float]:
    profile_text, career_text, skill_text, all_text = build_texts(candidate)
    archetype_value, archetype = archetype_score(profile_text)
    semantic = semantic_score(all_text)
    title = title_score(norm_text(candidate["profile"].get("current_title", "")))
    skill, expert_near_zero, expert_under_year = skill_depth_score(candidate)
    career, service_only = career_evidence_score(candidate, career_text)
    behavior = behavior_score(candidate["redrob_signals"])
    logistics = logistics_score(candidate["profile"], candidate["redrob_signals"])
    trust = trust_score(candidate, candidate["redrob_signals"])
    
    # Extract the 12 new features
    ret_depth = retrieval_depth_score(career_text)
    prod_ev = production_evidence_score(career_text)
    eval_fr = eval_framework_score(career_text)
    vec_db = vector_db_exp_score(all_text)
    
    years = float(candidate["profile"].get("years_of_experience", 0.0) or 0.0)
    exp_fit = experience_band_fit_score(years)
    
    last_active = parse_date(candidate["redrob_signals"].get("last_active_date"))
    if last_active is None:
        recency = 0.0
    else:
        days = max(0, (EVAL_DATE - last_active).days)
        if days <= 14:
            recency = 1.00
        elif days <= 30:
            recency = 0.88
        elif days <= 60:
            recency = 0.70
        elif days <= 120:
            recency = 0.42
        else:
            recency = 0.15
            
    hiring_ready = hiring_readiness_score(candidate["redrob_signals"])
    career_stab = career_stability_score(candidate)
    skill_cred = skill_credibility_score(expert_near_zero, expert_under_year)
    role_sem = archetype_value * semantic
    comp_qual = company_quality_score(candidate)
    nlp_ir = nlp_ir_focus_score(candidate)
    
    risk, hard_trap = risk_score(
        candidate,
        profile_text,
        career_text,
        all_text,
        archetype,
        semantic,
        title,
        service_only,
        expert_near_zero,
        expert_under_year,
    )
    
    rel = pseudo_relevance(
        archetype=archetype,
        semantic=semantic,
        title=title,
        skill=skill,
        career=career,
        behavior=behavior,
        logistics=logistics,
        trust=trust,
        retrieval_depth=ret_depth,
        production_evidence=prod_ev,
        eval_framework=eval_fr,
        vector_db_exp=vec_db,
        experience_band_fit=exp_fit,
        hiring_readiness=hiring_ready,
        career_stability=career_stab,
        skill_credibility=skill_cred,
        company_quality=comp_qual,
        nlp_ir_focus=nlp_ir,
        risk=risk,
        hard_trap=hard_trap
    )
    
    features = [
        archetype_value,
        semantic,
        title,
        career,
        skill,
        behavior,
        logistics,
        trust,
        ret_depth,
        prod_ev,
        eval_fr,
        vec_db,
        exp_fit,
        recency,
        hiring_ready,
        career_stab,
        skill_cred,
        role_sem,
        comp_qual,
        nlp_ir
    ]
    return features, risk, hard_trap, rel


def dcg(rels: np.ndarray, k: int) -> float:
    gains = np.power(2.0, rels[:k].astype(np.float64)) - 1.0
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2, dtype=np.float64))
    return float(np.sum(gains * discounts))


def average_precision(rels: np.ndarray, k: int, threshold: int = 3) -> float:
    hits = 0
    total = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        if rel >= threshold:
            hits += 1
            total += hits / i
    return total / max(1, hits)


def objective_for_scores(scores: np.ndarray, labels: np.ndarray, hard_traps: np.ndarray) -> float:
    n = len(scores)
    top_k = min(250, n)
    idx = np.argpartition(-scores, top_k - 1)[:top_k]
    idx = idx[np.argsort(-scores[idx])]
    ranked_rels = labels[idx]

    ideal = np.sort(labels)[::-1]
    idcg10 = max(dcg(ideal, min(10, n)), 1e-9)
    idcg50 = max(dcg(ideal, min(50, n)), 1e-9)
    ndcg10 = dcg(ranked_rels, min(10, top_k)) / idcg10
    ndcg50 = dcg(ranked_rels, min(50, top_k)) / idcg50
    ap = average_precision(ranked_rels, min(100, top_k), threshold=3)
    p10 = float(np.mean(ranked_rels[: min(10, top_k)] >= 3))
    trap_rate = float(np.mean(hard_traps[idx[: min(100, top_k)]]))

    return 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * ap + 0.05 * p10 - 0.35 * trap_rate


def ranking_metrics(scores: np.ndarray, labels: np.ndarray, hard_traps: np.ndarray, k: int = 100) -> Dict[str, float]:
    n = len(scores)
    k = min(k, n)
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    ranked_rels = labels[idx]
    ideal = np.sort(labels)[::-1]
    idcg = max(dcg(ideal, k), 1e-9)
    ndcg = dcg(ranked_rels, k) / idcg
    ap = average_precision(ranked_rels, k, threshold=3)
    p10 = float(np.mean(ranked_rels[: min(10, k)] >= 3))
    trap_rate = float(np.mean(hard_traps[idx]))
    return {
        "ndcg@100": float(ndcg),
        "ap@100": float(ap),
        "p@10": p10,
        "hard_trap_rate@100": trap_rate,
        "selection_metric": float(0.75 * ndcg + 0.25 * ap),
    }


def stratified_split(labels: np.ndarray, seed: int, valid_fraction: float = 0.20) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    binned_labels = np.round(labels).astype(np.int32)
    _, counts = np.unique(binned_labels, return_counts=True)
    if importlib.util.find_spec("sklearn") is not None and len(counts) > 1 and int(counts.min()) >= 2:
        from sklearn.model_selection import train_test_split

        train_idx, valid_idx = train_test_split(
            indices,
            test_size=valid_fraction,
            random_state=seed,
            stratify=binned_labels,
        )
        return np.asarray(train_idx, dtype=np.int64), np.asarray(valid_idx, dtype=np.int64)

    rng = np.random.default_rng(seed)
    train_parts: List[np.ndarray] = []
    valid_parts: List[np.ndarray] = []
    for label in np.unique(binned_labels):
        label_idx = indices[binned_labels == label].copy()
        rng.shuffle(label_idx)
        valid_count = max(1, int(round(len(label_idx) * valid_fraction)))
        valid_parts.append(label_idx[:valid_count])
        train_parts.append(label_idx[valid_count:])
    return np.concatenate(train_parts), np.concatenate(valid_parts)


def make_model(model_name: str, seed: int, trials: int, num_classes: int) -> Any:
    n_estimators = max(120, min(420, int(trials)))
    n_jobs = max(1, min(4, os.cpu_count() or 1))

    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=31,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=2.0,
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=-1,
        )

    if model_name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=5,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=2.0,
            gamma=0.1,
            random_state=seed,
            n_jobs=n_jobs,
            tree_method="hist",
        )

    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=300,
        max_depth=9,
        min_samples_leaf=8,
        random_state=seed,
        n_jobs=n_jobs,
    )


def select_model_name() -> str:
    if importlib.util.find_spec("lightgbm") is not None:
        return "lightgbm"
    if importlib.util.find_spec("xgboost") is not None:
        return "xgboost"
    return "sklearn_random_forest"


def expected_relevance(estimator: Any, matrix: np.ndarray, class_values: np.ndarray) -> np.ndarray:
    return estimator.predict(matrix)


def normalized_importances(estimator: Any, feature_count: int) -> np.ndarray:
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return np.zeros(feature_count, dtype=np.float32)
    values = np.asarray(importances, dtype=np.float32)
    if values.size != feature_count:
        fixed = np.zeros(feature_count, dtype=np.float32)
        fixed[: min(feature_count, values.size)] = values[:feature_count]
        values = fixed
    total = float(values.sum())
    if total > 0:
        values = values / total
    return values


def augmented_matrix(features: np.ndarray, risk: np.ndarray, hard_traps: np.ndarray) -> np.ndarray:
    return np.column_stack([features, risk.astype(np.float32), hard_traps.astype(np.float32)]).astype(np.float32)


def choose_posthoc_penalty(
    raw_scores: np.ndarray,
    risk: np.ndarray,
    hard_traps: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float, Dict[str, Any]]:
    raw_metrics = ranking_metrics(raw_scores, labels, hard_traps, k=100)
    best_metric = raw_metrics["selection_metric"]
    best_risk_penalty = DEFAULT_RISK_PENALTY
    best_trap_penalty = DEFAULT_HARD_TRAP_PENALTY
    best_metrics = raw_metrics

    risk_grid = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]
    trap_grid = [0.0, 0.25, 0.45, 0.75, 1.00]
    for risk_penalty in risk_grid:
        for trap_penalty in trap_grid:
            candidate_scores = raw_scores - risk_penalty * risk - trap_penalty * hard_traps.astype(np.float32)
            metrics = ranking_metrics(candidate_scores, labels, hard_traps, k=100)
            metric = metrics["selection_metric"]
            if metric > best_metric + 1e-12:
                best_metric = metric
                best_risk_penalty = risk_penalty
                best_trap_penalty = trap_penalty
                best_metrics = metrics

    return best_risk_penalty, best_trap_penalty, {
        "raw_model": raw_metrics,
        "selected_model": best_metrics,
        "posthoc_winner": "with_penalties" if best_metric > raw_metrics["selection_metric"] + 1e-12 else "raw_model",
        "risk_penalty": best_risk_penalty,
        "hard_trap_penalty": best_trap_penalty,
    }


def tune_weights(
    features: np.ndarray,
    risk: np.ndarray,
    labels: np.ndarray,
    hard_traps: np.ndarray,
    trials: int,
    seed: int,
) -> Tuple[RankerModel, float, float]:
    """Legacy name retained for main() compatibility; now trains a regressor."""
    matrix = augmented_matrix(features, risk, hard_traps)
    train_idx, valid_idx = stratified_split(labels, seed, valid_fraction=0.20)
    class_values = np.asarray(sorted(np.unique(np.round(labels).astype(int))), dtype=np.float64)
    y_train = labels[train_idx]

    sample_weight = 1.0 + y_train * 2.0

    model_name = select_model_name()
    estimator = make_model(model_name, seed, trials, num_classes=len(class_values))
    try:
        estimator.fit(matrix[train_idx], y_train, sample_weight=sample_weight)
    except TypeError:
        estimator.fit(matrix[train_idx], y_train)

    valid_raw_scores = expected_relevance(estimator, matrix[valid_idx], class_values)
    risk_penalty, trap_penalty, validation_metrics = choose_posthoc_penalty(
        valid_raw_scores,
        risk[valid_idx],
        hard_traps[valid_idx],
        labels[valid_idx],
    )
    feature_importances = normalized_importances(estimator, len(AUGMENTED_FEATURE_NAMES))
    validation_score = validation_metrics["selected_model"]["selection_metric"]
    validation_metrics["split"] = {
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(valid_idx)),
        "validation_fraction": 0.20,
        "stratified_on": "pseudo_label_rel",
    }
    validation_metrics["target_note"] = (
        "Validation metrics measure agreement with pseudo labels from extract_features(), "
        "not hidden Redrob ground truth."
    )

    ranker = RankerModel(
        estimator=estimator,
        model_name=model_name,
        class_values=class_values,
        feature_names=AUGMENTED_FEATURE_NAMES,
        feature_importances=feature_importances,
        risk_penalty=float(risk_penalty),
        hard_trap_penalty=float(trap_penalty),
        validation_metrics=validation_metrics,
    )
    return ranker, float(risk_penalty), float(validation_score)


def top_indices(scores: np.ndarray, ids: Sequence[str], top_n: int) -> List[int]:
    n = len(scores)
    pool = min(max(top_n * 4, 300), n)
    idx = np.argpartition(-scores, pool - 1)[:pool]
    return sorted(idx.tolist(), key=lambda i: (-float(scores[i]), ids[i]))[:top_n]


def evidence_terms(candidate: dict, limit: int = 5) -> List[str]:
    _, _, _, all_text = build_texts(candidate)
    terms: List[str] = []
    for needle, label in REASON_TERMS:
        if needle in all_text and label not in terms:
            terms.append(label)
        if len(terms) >= limit:
            break
    return terms


def relevant_skills(candidate: dict, limit: int = 4) -> List[str]:
    scored: List[Tuple[float, str]] = []
    for skill in candidate.get("skills", []):
        name = skill.get("name", "")
        normalized = norm_text(name)
        base = 0.0
        for term, weight in SKILL_WEIGHTS.items():
            if term in normalized:
                base = max(base, weight)
        if base <= 0:
            continue
        prof = {"beginner": 0.25, "intermediate": 0.55, "advanced": 0.82, "expert": 1.00}.get(
            skill.get("proficiency", "beginner"), 0.30
        )
        duration = clamp(numeric_value(skill, "duration_months", 0.0) / 60.0)
        endorsements = clamp(math.log1p(numeric_value(skill, "endorsements", 0.0)) / math.log1p(60))
        scored.append((base * prof * (0.7 + 0.3 * duration) * (0.8 + 0.2 * endorsements), name))
    return [name for _, name in sorted(scored, reverse=True)[:limit]]


def concern_fragments(candidate: dict) -> List[str]:
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    concerns: List[str] = []

    years = float(profile.get("years_of_experience", 0.0) or 0.0)
    if years < 5.0:
        concerns.append(f"{years:.1f} years is below the 5-9 year target")
    elif years > 9.0:
        concerns.append(f"{years:.1f} years is above the preferred 5-9 year band")

    notice = notice_days(signals)
    if notice >= 90:
        concerns.append(f"{notice}-day notice period")
    elif notice > 30:
        concerns.append(f"{notice}-day notice period is slower than ideal")

    response = numeric_value(signals, "recruiter_response_rate", 0.0)
    if response < 0.25:
        concerns.append(f"{response:.2f} recruiter response rate")

    country = norm_text(profile.get("country", ""))
    location = norm_text(profile.get("location", ""))
    if country != "india":
        if signals.get("willing_to_relocate"):
            concerns.append("outside India but willing to relocate")
        else:
            concerns.append("outside India with no relocation signal")
    elif not contains_any(location, PREFERRED_CITY_TERMS) and not signals.get("willing_to_relocate"):
        concerns.append("not in a preferred city and no relocation signal")

    return concerns


def positive_fragments(candidate: dict) -> List[str]:
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    positives: List[str] = []

    notice = notice_days(signals)
    if signals.get("open_to_work_flag") and notice <= 30:
        positives.append(f"open to work with {notice}-day notice")
    elif signals.get("open_to_work_flag"):
        positives.append("open to work")
    elif notice <= 30:
        positives.append(f"{notice}-day notice")

    response = numeric_value(signals, "recruiter_response_rate", 0.0)
    if response >= 0.50:
        positives.append(f"{response:.2f} recruiter response rate")

    saved = int(numeric_value(signals, "saved_by_recruiters_30d", 0.0))
    if saved >= 8:
        positives.append(f"{saved} recruiter saves in 30 days")

    github = numeric_value(signals, "github_activity_score", -1.0)
    if github >= 70:
        positives.append(f"{github:.1f} GitHub activity score")

    location = profile.get("location", "")
    if contains_any(norm_text(location), ["pune", "noida"]):
        positives.append(f"{location} location matches Pune/Noida preference")
    elif profile.get("country") == "India" and candidate["redrob_signals"].get("willing_to_relocate"):
        positives.append("India-based and willing to relocate")

    return positives


def make_reason(candidate: dict, rank: int) -> str:
    profile = candidate["profile"]
    years = float(profile.get("years_of_experience", 0.0) or 0.0)
    title = profile.get("current_title", "candidate")
    company = profile.get("current_company", "current company")
    location = profile.get("location", "unknown location")

    terms = evidence_terms(candidate, limit=5)
    skills = relevant_skills(candidate, limit=3)
    positives = positive_fragments(candidate)
    concerns = concern_fragments(candidate)

    if terms:
        evidence = ", ".join(terms[:5])
    elif skills:
        evidence = "skills in " + ", ".join(skills[:3])
    else:
        evidence = "adjacent AI/ML evidence"

    if rank <= 25:
        lead = "Strong fit"
    elif rank <= 70:
        lead = "Good fit"
    else:
        lead = "Relevant but lower-ranked"

    skill_clause = ""
    if skills:
        skill_clause = f"; listed skills include {', '.join(skills[:3])}"

    reason = (
        f"{lead}: {years:.1f} years as {title} at {company} in {location}, "
        f"with profile evidence for {evidence}{skill_clause}."
    )

    tail_items = positives[:2]
    if concerns:
        tail_items.append("concern: " + concerns[0])
    if tail_items:
        reason += " " + "; ".join(tail_items) + "."
    return reason


def load_feature_matrix(candidates_path: Path, verbose: bool = False) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids: List[str] = []
    rows: List[List[float]] = []
    risks: List[float] = []
    hard_traps: List[bool] = []
    labels: List[float] = []

    for i, candidate in enumerate(open_candidates(candidates_path), start=1):
        features, risk, hard_trap, rel = extract_features(candidate)
        ids.append(candidate["candidate_id"])
        rows.append(features)
        risks.append(risk)
        hard_traps.append(hard_trap)
        labels.append(rel)
        if verbose and i % 25000 == 0:
            print(f"Processed {i} candidates...", flush=True)

    return (
        ids,
        np.asarray(rows, dtype=np.float32),
        np.asarray(risks, dtype=np.float32),
        np.asarray(hard_traps, dtype=bool),
        np.asarray(labels, dtype=np.float32),
    )


def collect_selected_candidates(candidates_path: Path, selected_ids: set[str]) -> Dict[str, dict]:
    found: Dict[str, dict] = {}
    for candidate in open_candidates(candidates_path):
        cid = candidate["candidate_id"]
        if cid in selected_ids:
            found[cid] = candidate
            if len(found) == len(selected_ids):
                break
    return found


def write_submission(
    out_path: Path,
    ordered_ids: Sequence[str],
    raw_scores: Sequence[float],
    candidate_records: Dict[str, dict],
) -> None:
    max_score = max(raw_scores)
    min_score = min(raw_scores)
    denom = max(max_score - min_score, 1e-9)
    scaled = [0.35 + 0.64 * ((score - min_score) / denom) for score in raw_scores]

    # Enforce strictly decreasing printed scores to avoid validator tie issues
    # after decimal formatting.
    adjusted: List[float] = []
    previous = 1.01
    for value in scaled:
        value = min(value, previous - 1e-7)
        adjusted.append(value)
        previous = value

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, score) in enumerate(zip(ordered_ids, adjusted), start=1):
            writer.writerow([cid, rank, f"{score:.9f}", make_reason(candidate_records[cid], rank)])


def write_diagnostics(
    path: Path,
    ranker: RankerModel,
    penalty: float,
    validation_score: float,
    ordered_ids: Sequence[str],
    scores: Sequence[float],
    labels: np.ndarray,
    risks: np.ndarray,
    hard_traps: np.ndarray,
    id_to_index: Dict[str, int],
) -> None:
    selected_metrics = ranker.validation_metrics["selected_model"]
    raw_metrics = ranker.validation_metrics["raw_model"]
    split = ranker.validation_metrics["split"]
    lines = [
        "# Redrob Ranker Diagnostics",
        "",
        "model_type: supervised_tree_regressor",
        f"estimator: {ranker.model_name}",
        "target: pseudo_label_rel",
        f"validation_selection_metric: {validation_score:.6f}",
        f"risk_penalty: {penalty:.6f}",
        f"hard_trap_penalty: {ranker.hard_trap_penalty:.6f}",
        f"posthoc_penalty_winner: {ranker.validation_metrics['posthoc_winner']}",
        "",
        "validation_split:",
        f"  train_rows: {split['train_rows']}",
        f"  validation_rows: {split['validation_rows']}",
        f"  validation_fraction: {split['validation_fraction']:.2f}",
        f"  stratified_on: {split['stratified_on']}",
        "",
        "validation_metrics:",
        f"  raw_model_ndcg@100: {raw_metrics['ndcg@100']:.6f}",
        f"  raw_model_ap@100: {raw_metrics['ap@100']:.6f}",
        f"  raw_model_p@10: {raw_metrics['p@10']:.6f}",
        f"  raw_model_hard_trap_rate@100: {raw_metrics['hard_trap_rate@100']:.6f}",
        f"  selected_ndcg@100: {selected_metrics['ndcg@100']:.6f}",
        f"  selected_ap@100: {selected_metrics['ap@100']:.6f}",
        f"  selected_p@10: {selected_metrics['p@10']:.6f}",
        f"  selected_hard_trap_rate@100: {selected_metrics['hard_trap_rate@100']:.6f}",
        "",
        "accuracy_caveat:",
        f"  {ranker.validation_metrics['target_note']}",
        "",
        "feature_importance:",
        "  note: Tree-based model importances, not linear weights.",
    ]
    for name, importance in sorted(
        zip(ranker.feature_names, ranker.feature_importances),
        key=lambda item: (-float(item[1]), item[0]),
    ):
        lines.append(f"  {name}: {float(importance):.6f}")
    lines.extend(["", "top_100_summary:"])
    top_labels = [int(np.round(labels[id_to_index[cid]])) for cid in ordered_ids]
    top_risks = [float(risks[id_to_index[cid]]) for cid in ordered_ids]
    top_traps = [bool(hard_traps[id_to_index[cid]]) for cid in ordered_ids]
    lines.append(f"  pseudo_label_counts: {dict((label, top_labels.count(label)) for label in sorted(set(top_labels), reverse=True))}")
    lines.append(f"  hard_traps_in_top_100: {sum(top_traps)}")
    lines.append(f"  max_risk_in_top_100: {max(top_risks):.4f}")
    lines.append("")
    lines.append("top_20:")
    for rank, cid in enumerate(ordered_ids[:20], start=1):
        idx = id_to_index[cid]
        lines.append(
            f"  {rank:02d}. {cid} score={scores[rank - 1]:.6f} "
            f"pseudo_label={labels[idx]:.4f} risk={float(risks[idx]):.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank Redrob candidates and produce a top-100 CSV.")
    parser.add_argument("--candidates", type=Path, default=Path("candidates.jsonl"), help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", type=Path, default=Path("submission.csv"), help="Output CSV path")
    parser.add_argument("--top-n", type=int, default=100, help="Number of candidates to output; challenge requires 100")
    parser.add_argument("--trials", type=int, default=192, help="AutoML weight-search trials")
    parser.add_argument("--seed", type=int, default=20260629, help="Deterministic search seed")
    parser.add_argument("--diagnostics", type=Path, default=Path("ranker_diagnostics.md"), help="Diagnostics markdown path")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_n != 100:
        raise SystemExit("The Redrob submission spec requires --top-n 100.")
    if not args.candidates.exists():
        raise SystemExit(f"Cannot find candidates file: {args.candidates}")

    ids, features, risks, hard_traps, labels = load_feature_matrix(args.candidates, verbose=args.verbose)
    ranker, penalty, validation_score = tune_weights(features, risks, labels, hard_traps, args.trials, args.seed)
    model_scores = expected_relevance(ranker.estimator, augmented_matrix(features, risks, hard_traps), ranker.class_values)
    final_scores = model_scores - penalty * risks - ranker.hard_trap_penalty * hard_traps.astype(np.float32)
    top = top_indices(final_scores, ids, args.top_n)
    ordered_ids = [ids[i] for i in top]
    ordered_scores = [float(final_scores[i]) for i in top]

    records = collect_selected_candidates(args.candidates, set(ordered_ids))
    missing = [cid for cid in ordered_ids if cid not in records]
    if missing:
        raise SystemExit(f"Could not reload selected candidate records: {missing[:5]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_submission(args.out, ordered_ids, ordered_scores, records)

    id_to_index = {cid: i for i, cid in enumerate(ids)}
    write_diagnostics(args.diagnostics, ranker, penalty, validation_score, ordered_ids, ordered_scores, labels, risks, hard_traps, id_to_index)

    if args.verbose:
        print(f"Wrote {args.out}")
        print(f"Wrote {args.diagnostics}")


if __name__ == "__main__":
    main()
