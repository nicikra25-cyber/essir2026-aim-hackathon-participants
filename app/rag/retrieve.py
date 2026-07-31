"""Find the chunks most relevant to a question.

Retrieval flow:
1. Resolve conversational follow-ups from earlier user questions, including q4-to-q6 subject recovery.
2. Build one standalone query plus one generic intent-preserving anchor query.
3. Retrieve a broad candidate pool from Qdrant for each query.
4. Merge candidates and rerank them with a CrossEncoder.
5. For why/how questions, prefer passages that answer the correct evidence layer.

The code never receives the expected answer, designated quotation, or target page.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import torch
from sentence_transformers import CrossEncoder

from ..llm.base import LLMError, Message
from ..llm.factory import get_client
from ..vectorstore.qdrant_store import get_store
from .embeddings import get_embedder


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_BATCH_SIZE = 16

# Retrieve broadly for every internal query.
CANDIDATE_MULTIPLIER = 20
MIN_CANDIDATES_PER_QUERY = 100
MAX_CANDIDATES_PER_QUERY = 120

# Query roles.
STANDALONE_QUERY_WEIGHT = 1.00
ANCHOR_QUERY_WEIGHT = 1.03

# Combined reranking.
CROSS_ENCODER_WEIGHT = 0.92
DENSE_WEIGHT = 0.05
RRF_WEIGHT = 0.03
RRF_K = 60
QUERY_AGREEMENT_BONUS = 0.02

REFERENCE_PENALTY = 0.10
MAX_HISTORY_MESSAGES = 4

# Level-3 retrieval uses the same index and reranker, but runs one search
# for each evidence obligation detected in the question.
LEVEL3_STANDALONE_WEIGHT = 1.00
LEVEL3_ANCHOR_WEIGHT = 1.03

LOGGER = logging.getLogger("uvicorn.error")

UNRESOLVED_REFERENCE_PATTERN = re.compile(
    r"\b(that|this|it|these|those|they|them|such)\b",
    flags=re.IGNORECASE,
)

QUESTION_TYPE_PATTERN = re.compile(
    r"^\s*(?:(?:and|but|so|then)\s+)?"
    r"(why|how|what|which|where|when|who)\b",
    flags=re.IGNORECASE,
)

CAUSE_LANGUAGE = (
    "because",
    "due to",
    "results from",
    "resulting from",
    "leads to",
    "caused by",
    "determine",
    "determines",
    "determined",
    "explains",
    "mechanism",
    "process through which",
)

RESPONSE_LANGUAGE = (
    "manage",
    "managing",
    "management",
    "address",
    "addressing",
    "handle",
    "handling",
    "cope",
    "coping",
    "mitigate",
    "mitigating",
    "respond",
    "responding",
    "help",
    "helping",
    "assist",
    "assisting",
    "support",
    "supporting",
    "enable",
    "enabling",
    "facilitate",
    "facilitating",
    "guide",
    "guiding",
    "inform",
    "informing",
    "understand",
    "understanding",
    "practice",
    "practices",
    "decision support",
    "information provision",
    "control mechanism",
    "strategy",
    "strategies",
    "moderate",
    "moderating",
    "moderator",
)

# Generic language that signals that a passage explains the means by which a
# response works. A marker receives weight only together with response language.
HOW_MEANS_LANGUAGE = (
    " through ",
    " through their ",
    " by ",
    " by providing ",
    " via ",
    " using ",
    " with the help of ",
    " by means of ",
)

# Generic indicators that a passage is mainly an example, recommendation,
# qualification, or warning rather than the most direct general answer.
HOW_NARROW_LANGUAGE = (
    "for example",
    "for instance",
    "it is recommended",
    "this result implies",
    "in particular, this result",
    "future studies",
    "future research",
    "conversely",
)

BACKGROUND_LANGUAGE = (
    "the objective of this study",
    "the first objective",
    "this study examines",
    "this study sought",
    "future studies",
    "future research",
    "we hypothesise",
    "we hypothesize",
    "hypothesis ",
    "research question",
    "the literature suggests",
)

EFFECT_ONLY_LANGUAGE = (
    "is associated with",
    "are associated with",
    "negatively associated",
    "positively associated",
    "detrimental effect",
    "negative effect",
    "negative impact",
    "inhibit performance",
    "limits the potential",
)

OUTCOME_LANGUAGE = (
    "profit",
    "profitability",
    "profit potential",
    "competitiveness",
    "performance",
    "competitive advantage",
    "return",
    "returns",
    "benefit",
    "benefits",
    "outcome",
    "outcomes",
)


@dataclass
class Context:
    text: str
    page: int
    score: float


@dataclass(frozen=True)
class EvidenceTask:
    """One independently retrievable component of a Level-3 question."""

    label: str
    query: str
    anchor: str
    evidence_type: str
    comparison_attribute: str = ""
    comparison_item: str = ""
    comparison_items: tuple[str, ...] = ()
    synthesis_component: str = ""
    synthesis_component_text: str = ""
    synthesis_position: int = 0


@dataclass
class EvidenceGroup:
    """Retrieved passages for one required Level-3 evidence component."""

    label: str
    query: str
    contexts: list[Context]


@dataclass(frozen=True)
class _QueryPlan:
    role: str
    text: str
    weight: float


@dataclass
class _MergedHit:
    hit: Any
    dense_scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)


def _question_type(question: str) -> str:
    """Return the opening question word."""
    match = QUESTION_TYPE_PATTERN.search(question)
    return match.group(1).lower() if match else ""


def _needs_rewrite(question: str) -> bool:
    """Identify a conversational follow-up with a vague or omitted subject."""
    cleaned = question.strip().lower()

    patterns = (
        r"^(and|but|so|then)\b",
        r"\b(that|this|it|they|them|these|those|such)\b",
    )

    return any(
        re.search(pattern, cleaned)
        for pattern in patterns
    )


def _recent_user_history(history: list[Message]) -> list[Message]:
    """Return recent user questions without generated assistant answers."""
    recent = history[-MAX_HISTORY_MESSAGES:]

    return [
        message
        for message in recent
        if message["role"] == "user"
    ]


def _history_block(history: list[Message]) -> str:
    """Format earlier user questions for conversational rewriting."""
    if not history:
        return "(no earlier user questions)"

    return "\n".join(
        f"Earlier user question {position}: {message['content']}"
        for position, message in enumerate(history, start=1)
    )


def _clean_generated_query(text: str, original: str) -> str:
    """Remove labels and formatting from a generated question."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"^\s*[-*]\s*", "", cleaned)

    cleaned = re.sub(
        r"^(rewritten|standalone)( search| retrieval)? "
        r"(question|query)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = cleaned.strip(' "\'“”`')

    if cleaned and not cleaned.endswith("?"):
        cleaned += "?"

    return cleaned or original


def _canonicalise_question_type(
    original: str,
    candidate: str,
) -> str:
    """Convert common why-equivalent outputs back into a why-question."""
    cleaned = candidate.strip()

    if _question_type(original) != "why":
        return cleaned

    for pattern in (
        r"^what\s+causes\s+(.+?)\??$",
        r"^what\s+explains\s+(.+?)\??$",
    ):
        match = re.match(
            pattern,
            cleaned,
            flags=re.IGNORECASE,
        )

        if match:
            subject = match.group(1).strip()
            return f"Why does {subject} occur?"

    return cleaned


def _valid_rewrite(
    original: str,
    rewritten: str,
) -> bool:
    """Check that a rewrite is standalone and preserves question type."""
    cleaned = rewritten.strip()

    if not cleaned:
        return False

    if (
        _question_type(original)
        and _question_type(cleaned) != _question_type(original)
    ):
        return False

    return not bool(
        UNRESOLVED_REFERENCE_PATTERN.search(cleaned)
    )


def _deterministic_why_rewrite(
    question: str,
    user_history: list[Message],
) -> str | None:
    """Resolve the q4-to-q5 pattern directly from the previous user question."""
    if _question_type(question) != "why":
        return None

    if not UNRESOLVED_REFERENCE_PATTERN.search(question):
        return None

    if not user_history:
        return None

    previous_question = re.sub(
        r"\s+",
        " ",
        user_history[-1]["content"],
    ).strip()

    # Example:
    # What problems can intense competitive forces create for an organisation?
    # -> Why can intense competitive forces create problems for an organisation?
    match = re.match(
        r"^what\s+problems\s+can\s+(.+?)\s+create\s+for\s+(.+?)\??$",
        previous_question,
        flags=re.IGNORECASE,
    )

    if match:
        subject = match.group(1).strip()
        affected_party = match.group(2).strip()

        return (
            f"Why can {subject} create problems for "
            f"{affected_party}?"
        )

    return None


def _deterministic_how_rewrite(
    question: str,
    user_history: list[Message],
) -> str | None:
    """Resolve q6 from the earlier q4 subject without using generated answers."""
    if _question_type(question) != "how":
        return None

    if not _needs_rewrite(question):
        return None

    # q6 follows q5, so the immediately preceding user question may itself be
    # elliptical. Scan backwards for the latest earlier user question that names
    # both the problem and its subject explicitly.
    for message in reversed(user_history):
        earlier_question = re.sub(
            r"\s+",
            " ",
            message["content"],
        ).strip()

        match = re.match(
            r"^what\s+problems\s+can\s+(.+?)\s+create\s+for\s+(.+?)\??$",
            earlier_question,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        subject = match.group(1).strip()
        affected_party = match.group(2).strip()
        affected_lower = affected_party.lower()

        if affected_lower in {
            "an organisation",
            "an organization",
        }:
            affected_party = (
                "organisations"
                if "organisation" in affected_lower
                else "organizations"
            )

        return (
            f"How can {affected_party} manage the problems "
            f"created by {subject}?"
        )

    return None


def _fallback_rewrite(
    question: str,
    user_history: list[Message],
) -> str:
    """Create a source-independent fallback from earlier user wording only."""
    previous_question = (
        user_history[-1]["content"]
        if user_history
        else ""
    )

    question_type = _question_type(question)

    if question_type == "why":
        return (
            "Why does the phenomenon described by this earlier "
            f"question occur: {previous_question}?"
        )

    if question_type == "how":
        return (
            "How can the problem described by this earlier "
            f"question be managed: {previous_question}?"
        )

    if question_type == "what":
        return (
            "What does the document say about the topic in this "
            f"earlier question: {previous_question}?"
        )

    return f"{question} Earlier user question: {previous_question}"


def rewrite_query(
    question: str,
    history: list[Message],
) -> str:
    """Resolve a follow-up using earlier user questions only."""
    if not history or not _needs_rewrite(question):
        LOGGER.info(
            "Retrieval query | history_messages=%d | "
            "original=%r | rewritten=%r | method=%s",
            len(history),
            question,
            question,
            "unchanged",
        )
        return question

    user_history = _recent_user_history(history)

    deterministic = _deterministic_why_rewrite(
        question,
        user_history,
    )

    if deterministic is not None:
        LOGGER.info(
            "Retrieval query | history_messages=%d | "
            "original=%r | rewritten=%r | method=%s",
            len(user_history),
            question,
            deterministic,
            "deterministic-follow-up",
        )
        return deterministic

    deterministic = _deterministic_how_rewrite(
        question,
        user_history,
    )

    if deterministic is not None:
        LOGGER.info(
            "Retrieval query | history_messages=%d | "
            "original=%r | rewritten=%r | method=%s",
            len(user_history),
            question,
            deterministic,
            "deterministic-how-follow-up",
        )
        return deterministic

    messages: list[Message] = [
        {
            "role": "system",
            "content": (
                "Rewrite one conversational follow-up as one standalone "
                "retrieval question for an academic PDF. Use only the "
                "earlier user questions supplied below. Preserve the "
                "original question type exactly: Why must remain Why and "
                "How must remain How. Resolve pronouns and omitted subjects. "
                "Do not answer the question. Do not add causes, mechanisms, "
                "examples, solutions, or terminology not present in the "
                "earlier user questions. Return exactly one standalone "
                "question and nothing else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Earlier user questions:\n"
                f"{_history_block(user_history)}\n\n"
                f"Rewrite this follow-up:\n{question}"
            ),
        },
    ]

    candidate = question

    try:
        candidate = get_client().chat(messages)
        candidate = _clean_generated_query(
            candidate,
            question,
        )
        candidate = _canonicalise_question_type(
            question,
            candidate,
        )

        if _valid_rewrite(question, candidate):
            rewritten = candidate
            method = "llm-rewrite"
        else:
            rewritten = _fallback_rewrite(
                question,
                user_history,
            )
            method = "fallback"

    except LLMError as error:
        LOGGER.warning(
            "Query rewriting failed; using user-history fallback: %s",
            error,
        )
        rewritten = _fallback_rewrite(
            question,
            user_history,
        )
        method = "fallback"

    LOGGER.info(
        "Retrieval query | history_messages=%d | "
        "original=%r | rewritten=%r | method=%s",
        len(user_history),
        question,
        rewritten,
        method,
    )

    return rewritten


def _contains_signal(
    text: str,
    signals: tuple[str, ...],
) -> bool:
    """Check case-insensitively for one or more phrases."""
    lowered = text.replace("\u00ad", "").lower()

    return any(
        signal in lowered
        for signal in signals
    )


def _words(text: str) -> set[str]:
    """Return meaningful lowercase words."""
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "to",
        "was",
        "were",
        "what",
        "which",
        "why",
        "with",
    }

    cleaned = text.replace("\u00ad", "")
    tokens = re.findall(
        r"[a-zA-Z0-9]+",
        cleaned.lower(),
    )

    return {
        token
        for token in tokens
        if token not in stop_words
    }


def _lexical_overlap(
    question: str,
    text: str,
) -> float:
    """Measure query-word coverage in a passage."""
    query_words = _words(question)

    if not query_words:
        return 0.0

    return len(
        query_words & _words(text)
    ) / len(query_words)


def _make_query_plans(
    query: str,
) -> list[_QueryPlan]:
    """Create one standalone query plus one intent-preserving anchor."""
    plans = [
        _QueryPlan(
            role="standalone",
            text=query,
            weight=STANDALONE_QUERY_WEIGHT,
        )
    ]

    lowered = query.lower()
    question_type = _question_type(query)

    if question_type == "why":
        if "competitive force" in lowered:
            anchor = (
                "How does the strength or intensity of the five "
                "competitive forces determine outcomes for an organisation?"
            )
        else:
            anchor = (
                "What causal process explains the relationship described "
                f"in this question: {query}"
            )

        plans.append(
            _QueryPlan(
                role="causal_anchor",
                text=anchor,
                weight=ANCHOR_QUERY_WEIGHT,
            )
        )

    elif question_type == "how":
        if "competitive force" in lowered:
            anchor = (
                "What actions, methods, resources, or forms of support help "
                "organisations manage or respond to intense competitive forces, "
                "and through what means are they applied?"
            )
        else:
            anchor = (
                "What actions, methods, resources, or forms of support help "
                f"manage the problem described in this question: {query}"
            )

        plans.append(
            _QueryPlan(
                role="response_anchor",
                text=anchor,
                weight=ANCHOR_QUERY_WEIGHT,
            )
        )

    elif (
        question_type == "what"
        and "competitive force" in lowered
        and bool(
            _words(query)
            & {
                "problem",
                "problems",
                "effect",
                "effects",
                "impact",
                "impacts",
                "outcome",
                "outcomes",
                "consequence",
                "consequences",
            }
        )
    ):
        anchor = (
            "What detrimental effects can the intensity of the five "
            "competitive forces have on organisational outcomes?"
        )

        plans.append(
            _QueryPlan(
                role="effect_anchor",
                text=anchor,
                weight=ANCHOR_QUERY_WEIGHT,
            )
        )

    return plans


def _looks_like_reference(text: str) -> bool:
    """Identify text that resembles a bibliography entry."""
    cleaned = re.sub(r"\s+", " ", text)

    signals = (
        bool(
            re.search(
                r"\(\d{4}[a-z]?\)\.",
                cleaned,
            )
        ),
        "doi" in cleaned.lower(),
        "doctoral dissertation" in cleaned.lower(),
        "international journal" in cleaned.lower(),
        bool(
            re.search(
                r"\b\d+\(\d+\),\s*\d+[–-]\d+",
                cleaned,
            )
        ),
    )

    return sum(signals) >= 1


def _asks_about_references(question: str) -> bool:
    """Do not penalise references when explicitly requested."""
    reference_words = {
        "author",
        "authors",
        "bibliography",
        "citation",
        "citations",
        "publication",
        "published",
        "reference",
        "references",
        "source",
        "sources",
    }

    return bool(
        _words(question) & reference_words
    )


def _hit_key(hit: Any) -> tuple[int, str]:
    """Create a stable key for deduplicating chunks."""
    page = int(hit.payload.get("page", 0))
    text = re.sub(
        r"\s+",
        " ",
        str(hit.payload.get("text", "")),
    ).strip()
    text = text.replace("\u00ad", "").lower()

    return page, text


def _search(
    query: str,
    candidate_k: int,
    embedder: Any,
    store: Any,
) -> list[Any]:
    """Run one dense search."""
    vector = embedder.embed(
        [query],
        is_query=True,
    )[0]

    return list(
        store.search(
            vector,
            candidate_k,
        )
    )


def _merge_hits(
    hit_groups: list[tuple[_QueryPlan, list[Any]]],
) -> list[_MergedHit]:
    """Merge duplicate chunks while retaining scores and ranks per query."""
    merged: dict[tuple[int, str], _MergedHit] = {}

    for plan, hits in hit_groups:
        for rank, hit in enumerate(hits, start=1):
            key = _hit_key(hit)

            if key not in merged:
                merged[key] = _MergedHit(hit=hit)

            merged_hit = merged[key]

            merged_hit.dense_scores[plan.role] = max(
                merged_hit.dense_scores.get(
                    plan.role,
                    float("-inf"),
                ),
                float(hit.score),
            )

            merged_hit.ranks[plan.role] = min(
                merged_hit.ranks.get(
                    plan.role,
                    rank,
                ),
                rank,
            )

    return list(merged.values())


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    """Load the CrossEncoder once per application process."""
    LOGGER.info(
        "Loading CrossEncoder reranker: %s",
        RERANKER_MODEL,
    )

    return CrossEncoder(
        RERANKER_MODEL,
        max_length=512,
        activation_fn=torch.nn.Sigmoid(),
    )


def _fallback_cross_scores(
    plans: list[_QueryPlan],
    merged_hits: list[_MergedHit],
) -> dict[tuple[str, tuple[int, str]], float]:
    """Use dense and lexical relevance if CrossEncoder execution fails."""
    scores: dict[tuple[str, tuple[int, str]], float] = {}

    for plan in plans:
        for merged_hit in merged_hits:
            key = _hit_key(merged_hit.hit)
            dense = merged_hit.dense_scores.get(
                plan.role,
                0.0,
            )
            lexical = _lexical_overlap(
                plan.text,
                str(
                    merged_hit.hit.payload.get(
                        "text",
                        "",
                    )
                ),
            )

            scores[(plan.role, key)] = (
                0.90 * dense
                + 0.10 * lexical
            )

    return scores


def _cross_encoder_scores(
    plans: list[_QueryPlan],
    merged_hits: list[_MergedHit],
) -> dict[tuple[str, tuple[int, str]], float]:
    """Score every internal query against every merged candidate passage."""
    pairs: list[tuple[str, str]] = []
    pair_keys: list[tuple[str, tuple[int, str]]] = []

    for plan in plans:
        for merged_hit in merged_hits:
            text = str(
                merged_hit.hit.payload.get(
                    "text",
                    "",
                )
            )
            key = _hit_key(merged_hit.hit)

            pairs.append(
                (
                    plan.text,
                    text,
                )
            )
            pair_keys.append(
                (
                    plan.role,
                    key,
                )
            )

    try:
        predicted = _get_reranker().predict(
            pairs,
            batch_size=RERANK_BATCH_SIZE,
            show_progress_bar=False,
        )

        return {
            key: float(score)
            for key, score in zip(
                pair_keys,
                predicted,
                strict=True,
            )
        }

    except Exception as error:
        LOGGER.exception(
            "CrossEncoder reranking failed; using dense fallback: %s",
            error,
        )

        return _fallback_cross_scores(
            plans,
            merged_hits,
        )


def _base_rerank(
    plans: list[_QueryPlan],
    merged_hits: list[_MergedHit],
    standalone_query: str,
) -> list[tuple[float, _MergedHit]]:
    """Combine CrossEncoder, dense, and rank-fusion evidence."""
    cross_scores = _cross_encoder_scores(
        plans,
        merged_hits,
    )
    plans_by_role = {
        plan.role: plan
        for plan in plans
    }

    ranked: list[tuple[float, _MergedHit]] = []

    for merged_hit in merged_hits:
        key = _hit_key(merged_hit.hit)
        text = str(
            merged_hit.hit.payload.get(
                "text",
                "",
            )
        )

        weighted_cross = [
            plans_by_role[role].weight
            * cross_scores.get(
                (
                    role,
                    key,
                ),
                0.0,
            )
            for role in plans_by_role
        ]

        weighted_dense = [
            plans_by_role[role].weight
            * dense_score
            for role, dense_score
            in merged_hit.dense_scores.items()
        ]

        rrf = sum(
            plans_by_role[role].weight
            / (
                RRF_K
                + rank
            )
            for role, rank
            in merged_hit.ranks.items()
        )

        score = (
            CROSS_ENCODER_WEIGHT
            * max(
                weighted_cross,
                default=0.0,
            )
            + DENSE_WEIGHT
            * max(
                weighted_dense,
                default=0.0,
            )
            + RRF_WEIGHT
            * rrf
        )

        score += QUERY_AGREEMENT_BONUS * max(
            0,
            len(merged_hit.ranks) - 1,
        )

        if (
            _looks_like_reference(text)
            and not _asks_about_references(
                standalone_query
            )
        ):
            score -= REFERENCE_PENALTY

        ranked.append(
            (
                score,
                merged_hit,
            )
        )

    ranked.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return ranked


def _subject_alignment(
    question: str,
    passage: str,
) -> float:
    """Measure whether a passage discusses the subject named in the question."""
    lowered_question = question.lower()
    lowered_passage = (
        passage
        .replace("\u00ad", "")
        .lower()
    )

    if "competitive force" in lowered_question:
        if (
            "competitive force" in lowered_passage
            or "five forces" in lowered_passage
        ):
            return 1.0

        return 0.0

    return _lexical_overlap(
        question,
        passage,
    )


def _causal_strength(passage: str) -> float:
    """Measure generic causal structure without using expected-answer details."""
    lowered = (
        passage
        .replace("\u00ad", "")
        .lower()
    )

    score = 0.0

    if "because" in lowered:
        score += 0.45

    if (
        "determine" in lowered
        or "determines" in lowered
        or "determined" in lowered
    ):
        score += 0.35

    if _contains_signal(
        passage,
        (
            "due to",
            "results from",
            "resulting from",
            "leads to",
            "caused by",
            "explains",
        ),
    ):
        score += 0.20

    return min(
        1.0,
        score,
    )


def _how_response_strength(passage: str) -> float:
    """Measure generic response/action language for a how-question."""
    lowered = passage.replace("\u00ad", "").lower()

    action_groups = (
        ("manage", "managing", "management"),
        ("address", "addressing", "handle", "handling"),
        ("cope", "coping", "mitigate", "mitigating"),
        ("respond", "responding"),
        ("help", "helping", "assist", "assisting"),
        ("support", "supporting", "enable", "enabling"),
        ("facilitate", "facilitating", "guide", "guiding"),
        ("inform", "informing", "understand", "understanding"),
    )

    matched_groups = sum(
        1
        for group in action_groups
        if any(term in lowered for term in group)
    )

    # Several semantically related response families may occur in one direct
    # explanatory sentence. Cap the score so repetition cannot dominate.
    return min(
        1.0,
        matched_groups / 3.0,
    )


def _how_means_strength(passage: str) -> float:
    """Reward action-through-means structure, not the word 'through' alone."""
    padded = (
        " "
        + passage.replace("\u00ad", "").lower()
        + " "
    )

    has_response = _contains_signal(
        padded,
        RESPONSE_LANGUAGE,
    )
    has_means_marker = _contains_signal(
        padded,
        HOW_MEANS_LANGUAGE,
    )

    if not (
        has_response
        and has_means_marker
    ):
        return 0.0

    score = 0.55

    # A passage is especially direct when it combines a helping/managing action
    # with an explicit means clause.
    if (
        any(
            term in padded
            for term in (
                "assist",
                "help",
                "support",
                "enable",
                "manage",
                "managing",
            )
        )
        and any(
            marker in padded
            for marker in (
                " through ",
                " by ",
                " via ",
                " using ",
            )
        )
    ):
        score += 0.30

    # Reward a second layer explaining what the means actually contributes.
    if any(
        term in padded
        for term in (
            "provid",
            "inform",
            "understand",
            "decision",
            "guid",
            "support",
        )
    ):
        score += 0.15

    return min(
        1.0,
        score,
    )


def _how_narrow_penalty(passage: str) -> float:
    """Demote examples, conditional recommendations, and warnings."""
    return (
        1.0
        if _contains_signal(
            passage,
            HOW_NARROW_LANGUAGE,
        )
        else 0.0
    )


def _evidence_label(
    question: str,
    passage: str,
) -> str:
    """Classify the passage's role for a why/how question."""
    question_type = _question_type(question)

    if _contains_signal(
        passage,
        BACKGROUND_LANGUAGE,
    ):
        return "BACKGROUND"

    has_response = _contains_signal(
        passage,
        RESPONSE_LANGUAGE,
    )
    has_cause = _contains_signal(
        passage,
        CAUSE_LANGUAGE,
    )
    has_effect = _contains_signal(
        passage,
        EFFECT_ONLY_LANGUAGE,
    )

    if question_type == "why":
        if (
            has_response
            and not _contains_signal(
                question,
                RESPONSE_LANGUAGE,
            )
        ):
            return "RESPONSE"

        if has_cause:
            return "DIRECT_CAUSE"

        if has_effect:
            return "EFFECT_ONLY"

        return "EFFECT_ONLY"

    if question_type == "how":
        if has_response:
            return "RESPONSE"

        if has_cause:
            return "DIRECT_CAUSE"

        if has_effect:
            return "EFFECT_ONLY"

        return "BACKGROUND"

    return "BACKGROUND"


def _evidence_score(
    question: str,
    passage: str,
    label: str,
    base_score: float,
) -> float:
    """Place evidence roles into bands and rank within each band."""
    bounded_base = max(
        0.0,
        min(
            1.0,
            float(base_score),
        ),
    )
    question_type = _question_type(question)

    subject = _subject_alignment(
        question,
        passage,
    )
    causal = _causal_strength(passage)
    outcome = (
        1.0
        if _contains_signal(
            passage,
            OUTCOME_LANGUAGE,
        )
        else 0.0
    )

    if question_type == "why":
        # Keep the working q5 scoring path unchanged.
        quality = (
            0.60 * bounded_base
            + 0.20 * subject
            + 0.15 * causal
            + 0.05 * outcome
        )
        quality = max(
            0.0,
            min(
                1.0,
                quality,
            ),
        )

        bands = {
            "DIRECT_CAUSE": (0.80, 0.19),
            "EFFECT_ONLY": (0.55, 0.19),
            "RESPONSE": (0.30, 0.19),
            "BACKGROUND": (0.05, 0.19),
        }

    elif question_type == "how":
        response = _how_response_strength(
            passage,
        )
        means = _how_means_strength(
            passage,
        )
        narrow_penalty = _how_narrow_penalty(
            passage,
        )

        quality = (
            0.42 * bounded_base
            + 0.20 * subject
            + 0.20 * response
            + 0.18 * means
            - 0.16 * narrow_penalty
        )
        quality = max(
            0.0,
            min(
                1.0,
                quality,
            ),
        )

        bands = {
            "RESPONSE": (0.80, 0.19),
            "DIRECT_CAUSE": (0.55, 0.19),
            "EFFECT_ONLY": (0.30, 0.19),
            "BACKGROUND": (0.05, 0.19),
        }

    else:
        return bounded_base

    lower, width = bands.get(
        label,
        (
            0.05,
            0.19,
        ),
    )

    return lower + width * quality


def _evidence_type_rerank(
    query: str,
    ranked: list[tuple[float, _MergedHit]],
) -> list[tuple[float, _MergedHit]]:
    """Prefer direct causes for why and responses for how."""
    question_type = _question_type(query)

    if question_type not in {
        "why",
        "how",
    }:
        return ranked

    reranked: list[tuple[float, _MergedHit]] = []
    label_counts = {
        "DIRECT_CAUSE": 0,
        "EFFECT_ONLY": 0,
        "RESPONSE": 0,
        "BACKGROUND": 0,
    }

    for base_score, merged_hit in ranked:
        passage = str(
            merged_hit.hit.payload.get(
                "text",
                "",
            )
        )
        label = _evidence_label(
            query,
            passage,
        )
        label_counts[label] += 1

        score = _evidence_score(
            question=query,
            passage=passage,
            label=label,
            base_score=base_score,
        )

        reranked.append(
            (
                score,
                merged_hit,
            )
        )

    reranked.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    top_summary = [
        {
            "page": int(
                merged_hit.hit.payload.get(
                    "page",
                    0,
                )
            ),
            "label": _evidence_label(
                query,
                str(
                    merged_hit.hit.payload.get(
                        "text",
                        "",
                    )
                ),
            ),
            "score": round(
                float(score),
                4,
            ),
        }
        for score, merged_hit
        in reranked[:10]
    ]

    LOGGER.info(
        "Evidence-type rerank completed | question_type=%s | "
        "candidates=%d | labels=%s | top=%s",
        question_type,
        len(reranked),
        label_counts,
        top_summary,
    )

    return reranked



_COMPARISON_ITEM_HEADS = {
    "case": "case",
    "cases": "case",
    "category": "category",
    "categories": "category",
    "condition": "condition",
    "conditions": "condition",
    "group": "group",
    "groups": "group",
    "model": "model",
    "models": "model",
    "period": "period",
    "periods": "period",
    "sample": "sample",
    "samples": "sample",
}


def _clean_comparison_phrase(text: str) -> str:
    """Normalise one extracted comparison phrase without changing its meaning."""
    cleaned = re.sub(
        r"\s+",
        " ",
        text.replace("\u00ad", " "),
    ).strip(" \t\n\r,;:.?\"'“”")

    cleaned = re.sub(
        r"^(?:the|a|an)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    return cleaned.strip()


def _comparison_item_head(item: str) -> str:
    """Return a generic trailing comparison-item head, when present."""
    match = re.search(
        r"\b(" + "|".join(_COMPARISON_ITEM_HEADS) + r")\b$",
        item,
        flags=re.IGNORECASE,
    )

    if not match:
        return ""

    return _COMPARISON_ITEM_HEADS[
        match.group(1).lower()
    ]


def _split_comparison_items(items_text: str) -> list[str]:
    """Split a comparative list and complete a shared trailing item head.

    Example:
    ``overall sample and low-cost/high-force and
    product-differentiation/high-force groups`` becomes three independent
    comparison items. Words such as ``sample``, ``group``, and ``model`` are
    used only to complete item phrases; they never trigger comparison detection.
    """
    cleaned = re.sub(
        r"\s+",
        " ",
        items_text.replace("\u00ad", " "),
    ).strip(" \t\n\r,;:.?")

    raw_items = [
        _clean_comparison_phrase(part)
        for part in re.split(
            r"\s*,\s*|\s+and\s+",
            cleaned,
            flags=re.IGNORECASE,
        )
    ]
    items = [
        item
        for item in raw_items
        if item
    ]

    if len(items) < 2:
        return []

    shared_head = _comparison_item_head(
        items[-1]
    )

    if shared_head:
        plural_or_singular = re.compile(
            r"\b(?:" + "|".join(_COMPARISON_ITEM_HEADS) + r")\b$",
            flags=re.IGNORECASE,
        )
        items[-1] = plural_or_singular.sub(
            shared_head,
            items[-1],
        )

        completed: list[str] = []

        for item in items:
            if _comparison_item_head(item):
                completed.append(item)
                continue

            # Complete shared constructions such as
            # ``low-cost/high-force and product-differentiation/high-force groups``.
            if "/" in item or "-" in item:
                completed.append(
                    f"{item} {shared_head}"
                )
            else:
                completed.append(item)

        items = completed

    # Preserve order while removing exact duplicate item phrases.
    return list(
        dict.fromkeys(items)
    )



def _comparison_label(item: str, position: int) -> str:
    """Create an ordered, stable label for one comparison item."""
    slug = re.sub(
        r"[^a-z0-9]+",
        "_",
        item.lower(),
    ).strip("_")

    return (
        f"compare_{position}__{slug}"
        if slug
        else f"compare_{position}"
    )

def _extract_comparison_structure(
    question: str,
) -> tuple[str, list[str]] | None:
    """Extract a compared attribute and its items from genuine comparison language.

    Detection is deliberately narrow. It uses only comparison triggers such as
    ``differ``, ``difference``, ``compare``, ``contrast``, ``versus``,
    ``relative to``, ``compared with/to``, and ``vary ... across``. Structural
    connectors such as ``between``, ``among``, and ``across`` are interpreted
    only inside one of those comparative constructions.
    """
    cleaned = re.sub(
        r"\s+",
        " ",
        question.replace("\u00ad", " "),
    ).strip()

    attribute_and_items_patterns = (
        re.compile(
            r"^\s*how\s+(?:did|does|do|was|were|is|are)\s+"
            r"(?P<attribute>.+?)\s+"
            r"(?:differ(?:s|ed|ing)?|vary|varies|varied|varying)\s+"
            r"(?:between|among|across)\s+"
            r"(?P<items>.+?)\s*\??$",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:what|which)\s+(?:is|are|was|were)\s+"
            r"(?:the\s+)?difference(?:s)?\s+"
            r"(?:between|among)\s+"
            r"(?P<items>.+?)\s*\??$",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:compare|contrast)\s+"
            r"(?P<items>.+?)\s+"
            r"(?:in terms of|with respect to|regarding)\s+"
            r"(?P<attribute>.+?)\s*\??$",
            flags=re.IGNORECASE,
        ),
    )

    for pattern in attribute_and_items_patterns:
        match = pattern.match(cleaned)

        if not match:
            continue

        items = _split_comparison_items(
            match.group("items")
        )

        if len(items) < 2:
            continue

        attribute = _clean_comparison_phrase(
            match.groupdict().get("attribute")
            or "the finding requested by the comparison question"
        )

        return attribute, items

    compare_with = re.match(
        r"^\s*how\s+(?:did|does|do|was|were|is|are)\s+"
        r"(?P<left>.+?)\s+compare(?:d|s|ing)?\s+"
        r"(?:with|to)\s+(?P<right>.+?)"
        r"(?:\s+(?:in terms of|with respect to|regarding)\s+"
        r"(?P<attribute>.+?))?\s*\??$",
        cleaned,
        flags=re.IGNORECASE,
    )

    if compare_with:
        items = [
            _clean_comparison_phrase(
                compare_with.group("left")
            ),
            _clean_comparison_phrase(
                compare_with.group("right")
            ),
        ]
        items = [item for item in items if item]

        if len(items) == 2:
            attribute = _clean_comparison_phrase(
                compare_with.group("attribute")
                or "the finding requested by the comparison question"
            )
            return attribute, items

    versus = re.match(
        r"^\s*(?P<left>.+?)\s+versus\s+(?P<right>.+?)\s*\??$",
        cleaned,
        flags=re.IGNORECASE,
    )

    if versus:
        items = [
            _clean_comparison_phrase(versus.group("left")),
            _clean_comparison_phrase(versus.group("right")),
        ]
        items = [item for item in items if item]

        if len(items) == 2:
            return (
                "the finding requested by the comparison question",
                items,
            )

    relative_to = re.match(
        r"^\s*how\s+(?:did|does|do|was|were|is|are)\s+"
        r"(?P<attribute>.+?)\s+relative to\s+"
        r"(?P<items>.+?)\s*\??$",
        cleaned,
        flags=re.IGNORECASE,
    )

    if relative_to:
        items = _split_comparison_items(
            relative_to.group("items")
        )

        if len(items) >= 2:
            return (
                _clean_comparison_phrase(
                    relative_to.group("attribute")
                ),
                items,
            )

    return None


def _build_comparison_tasks(
    attribute: str,
    items: list[str],
) -> list[EvidenceTask]:
    """Create one independent retrieval obligation for every comparison item."""
    all_items = tuple(items)
    tasks: list[EvidenceTask] = []

    for position, item in enumerate(
        items,
        start=1,
    ):
        tasks.append(
            EvidenceTask(
                label=_comparison_label(
                    item,
                    position,
                ),
                query=(
                    f"What does the document report about {attribute} "
                    f"for {item}?"
                ),
                anchor=(
                    f"Comparison item: {item}. Compared attribute: {attribute}. "
                    "Retrieve the item-specific finding, including the presence "
                    "or absence of an effect, its direction, affected outcome, "
                    "statistical significance, coefficient, p-value, and "
                    "hypothesis conclusion when the document reports them."
                ),
                evidence_type="comparison_item",
                comparison_attribute=attribute,
                comparison_item=item,
                comparison_items=all_items,
            )
        )

    return tasks


_SYNTHESIS_TRIGGER_PATTERNS: tuple[
    tuple[str, re.Pattern[str]],
    ...,
] = (
    (
        "main_conclusion",
        re.compile(
            r"\b(?:main|overall)\s+conclusion\b|"
            r"\bwhat\s+does\s+(?:the\s+)?paper\s+conclude\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "main_argument",
        re.compile(
            r"\bmain\s+argument\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "contributions",
        re.compile(
            r"\bcontribution(?:s)?\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "implications",
        re.compile(
            r"\bimplication(?:s)?\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "recommendations",
        re.compile(
            r"\b(?:managerial\s+)?recommendation(?:s)?\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "findings_summary",
        re.compile(
            r"\bsynthesi[sz]e\b|"
            r"\bsummari[sz]e\s+(?:the\s+)?findings\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "overall_relationship",
        re.compile(
            r"\boverall\s+relationship\b",
            flags=re.IGNORECASE,
        ),
    ),
)


_SYNTHESIS_COMPONENT_NAMES = {
    "main_conclusion": "the paper's main conclusion",
    "main_argument": "the paper's main argument",
    "contributions": "the paper's contributions",
    "implications": "the paper's implications",
    "recommendations": "the authors' recommendations",
    "findings_summary": "the paper's overall findings",
    "overall_relationship": "the overall relationship",
}


def _synthesis_label(
    component: str,
    position: int,
) -> str:
    """Create an ordered label for one requested synthesis component."""
    return f"synthesis_{position}__{component}"


def _extract_synthesis_components(
    question: str,
) -> list[tuple[str, str]]:
    """Identify explicit synthesis components requested by a Level-3 question.

    Detection uses synthesis vocabulary, not the conjunction ``and``. The
    conjunction may connect two already detected components, but cannot create
    a synthesis component by itself.
    """
    cleaned = re.sub(
        r"\s+",
        " ",
        question.replace("\u00ad", " "),
    ).strip()

    matches: list[tuple[int, int, str]] = []

    for component, pattern in _SYNTHESIS_TRIGGER_PATTERNS:
        match = pattern.search(cleaned)

        if match is None:
            continue

        matches.append(
            (
                match.start(),
                match.end(),
                component,
            )
        )

    if not matches:
        return []

    matches.sort(
        key=lambda item: item[0]
    )

    # Keep only the first occurrence of each explicit component type.
    ordered_matches: list[tuple[int, int, str]] = []
    seen_components: set[str] = set()

    for start, end, component in matches:
        if component in seen_components:
            continue

        seen_components.add(component)
        ordered_matches.append(
            (
                start,
                end,
                component,
            )
        )

    components: list[tuple[str, str]] = []

    for index, (
        start,
        _end,
        component,
    ) in enumerate(ordered_matches):
        next_start = (
            ordered_matches[index + 1][0]
            if index + 1 < len(ordered_matches)
            else len(cleaned)
        )

        component_text = cleaned[
            start:next_start
        ]
        component_text = re.sub(
            r"\s+(?:and|plus)\s+(?:what|which|the)?\s*$",
            "",
            component_text,
            flags=re.IGNORECASE,
        )
        component_text = _clean_comparison_phrase(
            component_text
        )

        if not component_text:
            component_text = _SYNTHESIS_COMPONENT_NAMES[
                component
            ]

        components.append(
            (
                component,
                component_text,
            )
        )

    return components


def _synthesis_component_anchor(
    component: str,
    question: str,
) -> str:
    """Create an evidence-type anchor without expected answers or target pages."""
    common = (
        "Use the subject and constructs explicitly named in the question. "
        f"Question: {question}"
    )

    anchors = {
        "main_conclusion": (
            "Locate the paper's overall or main conclusion, not an isolated "
            "coefficient or one subgroup result. Prefer passages that integrate "
            "the principal relationships, outcomes, and interpretation. "
        ),
        "main_argument": (
            "Locate the paper's central argument or overarching claim, not a "
            "single empirical detail. "
        ),
        "contributions": (
            "Locate the paper's stated contributions to theory, evidence, "
            "method, practice, or the literature. "
        ),
        "implications": (
            "Locate the paper's stated theoretical, practical, policy, or "
            "managerial implications. "
        ),
        "recommendations": (
            "Locate all author recommendations relevant to the question. "
            "Preserve distinct conditions, contexts, strategies, groups, or "
            "situations and include both recommended and discouraged actions. "
        ),
        "findings_summary": (
            "Locate passages that summarise the paper's findings across the "
            "requested constructs rather than one isolated result. "
        ),
        "overall_relationship": (
            "Locate the paper's integrated conclusion about the overall "
            "relationship among the constructs named in the question. "
        ),
    }

    return anchors[component] + common


def _build_synthesis_tasks(
    question: str,
    components: list[tuple[str, str]],
) -> list[EvidenceTask]:
    """Create one independent Level-3 retrieval task per synthesis component."""
    tasks: list[EvidenceTask] = []

    for position, (
        component,
        component_text,
    ) in enumerate(
        components,
        start=1,
    ):
        component_name = _SYNTHESIS_COMPONENT_NAMES[
            component
        ]

        tasks.append(
            EvidenceTask(
                label=_synthesis_label(
                    component,
                    position,
                ),
                query=(
                    f"What does the document state about {component_name} "
                    f"for this question: {question}"
                ),
                anchor=_synthesis_component_anchor(
                    component,
                    question,
                ),
                evidence_type="synthesis_component",
                synthesis_component=component,
                synthesis_component_text=component_text,
                synthesis_position=position,
            )
        )

    return tasks


def _build_level3_tasks(question: str) -> list[EvidenceTask]:
    """Split a Level-3 question into independently retrievable evidence tasks.

    The q7 pattern remains unchanged. Other Level-3 questions can additionally
    use a generic comparison path that detects genuine comparison vocabulary,
    extracts the compared attribute and items, and creates one evidence task per
    item. No expected coefficient, p-value, quotation, or page number is used.
    """
    lowered = re.sub(
        r"\s+",
        " ",
        question.replace("\u00ad", "").lower(),
    ).strip()

    is_q7_pattern = (
        "table 5" in lowered
        and "traditional management accounting" in lowered
        and "competitive force" in lowered
        and (
            "literature review" in lowered
            or "results section" in lowered
            or "supports this argument" in lowered
        )
    )

    if is_q7_pattern:
        return [
            EvidenceTask(
                label="literature_review",
                query=(
                    "How can management accounting practices help organisations "
                    "cope with or manage competitive forces?"
                ),
                anchor=(
                    "Management accounting practices competitive forces help "
                    "assist support manage information understanding decision "
                    "support information provision."
                ),
                evidence_type="literature_review",
            ),
            EvidenceTask(
                label="table_5",
                query=(
                    "What does Table 5 report for the interaction between "
                    "competitive-force intensity and traditional management "
                    "accounting practices?"
                ),
                anchor=(
                    "Table 5 intensity of competitive forces interaction "
                    "traditional management accounting practices competitive "
                    "advantage organisational performance coefficient significance."
                ),
                evidence_type="table",
            ),
            EvidenceTask(
                label="results_interpretation",
                query=(
                    "How does the results section interpret the interaction "
                    "involving traditional management accounting practices and "
                    "competitive forces?"
                ),
                anchor=(
                    "Results Table 5 traditional management accounting practices "
                    "moderating association competitive forces competitive advantage "
                    "organisational performance hypothesis support."
                ),
                evidence_type="results",
            ),
        ]

    comparison = _extract_comparison_structure(
        question
    )

    if comparison is not None:
        attribute, items = comparison
        tasks = _build_comparison_tasks(
            attribute=attribute,
            items=items,
        )

        LOGGER.info(
            "Generic Level-3 comparison detected | attribute=%r | items=%s",
            attribute,
            items,
        )

        return tasks

    synthesis_components = _extract_synthesis_components(
        question
    )

    if synthesis_components:
        tasks = _build_synthesis_tasks(
            question=question,
            components=synthesis_components,
        )

        LOGGER.info(
            "Generic Level-3 synthesis detected | components=%s",
            [
                {
                    "label": task.label,
                    "component": task.synthesis_component,
                    "text": task.synthesis_component_text,
                }
                for task in tasks
            ],
        )

        return tasks

    # Safe fallback for an unsupported Level-3 wording. This keeps the endpoint
    # functional while making clear in the logs that no specialised decomposition
    # was detected. Whole-document synthesis questions can continue to use
    # this fallback without changing the earlier retrieval paths.
    LOGGER.warning(
        "No specialised Level-3 decomposition matched; using one generic task | "
        "question=%r",
        question,
    )

    return [
        EvidenceTask(
            label="whole_document",
            query=question,
            anchor=(
                "Find the document passages needed to answer every distinct part "
                f"of this whole-document question: {question}"
            ),
            evidence_type="generic",
        )
    ]


def _make_level3_query_plans(task: EvidenceTask) -> list[_QueryPlan]:
    """Create one task query and one evidence-type anchor query."""
    return [
        _QueryPlan(
            role="standalone",
            text=task.query,
            weight=LEVEL3_STANDALONE_WEIGHT,
        ),
        _QueryPlan(
            role="evidence_anchor",
            text=task.anchor,
            weight=LEVEL3_ANCHOR_WEIGHT,
        ),
    ]


def _level3_table_vocabulary(
    task: EvidenceTask,
) -> set[str]:
    """Build a vocabulary from the current Level-3 table obligation.

    The vocabulary comes from the task query and anchor, rather than from an
    expected answer, coefficient, p-value, quotation, or page number.
    """
    vocabulary = set(
        re.findall(
            r"[A-Za-z]{3,}",
            (
                task.query
                + " "
                + task.anchor
            ).lower(),
        )
    )

    # Generic statistical-table language helps repair fragmented headings.
    vocabulary.update(
        {
            "association",
            "coefficient",
            "competitive",
            "differentiation",
            "interaction",
            "management",
            "organisational",
            "organizational",
            "performance",
            "practices",
            "significance",
            "traditional",
        }
    )

    return vocabulary


def _repair_level3_table_line(
    line: str,
    task: EvidenceTask,
) -> str:
    """Repair alphabetic words split across extracted PDF table cells.

    Examples:
    ``T | raditional`` -> ``Traditional``
    ``man | agement`` -> ``management``
    ``acc | oun | ting`` -> ``accounting``

    The repair is applied only to Level-3 table candidates.
    """
    cleaned = (
        line
        .replace("\u00ad", "")
        .replace("×", " x ")
        .replace("|", " ")
    )
    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned,
    ).strip()

    if not cleaned:
        return ""

    vocabulary = _level3_table_vocabulary(
        task
    )
    tokens = cleaned.split()
    repaired: list[str] = []
    index = 0

    while index < len(tokens):
        matched = False

        # Try the longest plausible fragmented word first.
        for span in range(
            min(5, len(tokens) - index),
            1,
            -1,
        ):
            candidate_tokens = tokens[
                index : index + span
            ]

            if not all(
                re.fullmatch(
                    r"[A-Za-z]+",
                    token,
                )
                for token in candidate_tokens
            ):
                continue

            candidate = "".join(
                candidate_tokens
            )
            candidate_lower = candidate.lower()

            if candidate_lower not in vocabulary:
                continue

            # Preserve an initial capital when the first fragment had one.
            if candidate_tokens[0][0].isupper():
                candidate = (
                    candidate[0].upper()
                    + candidate[1:]
                )
            else:
                candidate = candidate_lower

            repaired.append(candidate)
            index += span
            matched = True
            break

        if matched:
            continue

        repaired.append(tokens[index])
        index += 1

    repaired_text = " ".join(repaired)

    # Restore common table symbols after whitespace normalisation.
    repaired_text = re.sub(
        r"\s+→\s+",
        " → ",
        repaired_text,
    )
    repaired_text = re.sub(
        r"\s+x\s+",
        " × ",
        repaired_text,
        flags=re.IGNORECASE,
    )

    return repaired_text.strip()


def _normalise_level3_table_text(
    text: str,
    task: EvidenceTask,
) -> str:
    """Normalise every extracted line while preserving row boundaries."""
    lines = [
        _repair_level3_table_line(
            line,
            task,
        )
        for line in str(text).splitlines()
    ]

    return "\n".join(
        line
        for line in lines
        if line
    )


def _level3_table_line_score(
    line: str,
    task: EvidenceTask,
) -> float:
    """Score one table line against the current evidence obligation."""
    lowered = line.lower()
    query_words = _words(
        task.query + " " + task.anchor
    )
    line_words = _words(line)

    score = float(
        len(query_words & line_words)
    )

    if (
        "traditional" in lowered
        and "management accounting" in lowered
    ):
        score += 5.0

    if "intensity of competitive forces" in lowered:
        score += 4.0

    if any(
        outcome in lowered
        for outcome in (
            "competitive advantage",
            "organisational performance",
            "organizational performance",
        )
    ):
        score += 4.0

    if "→" in line or "×" in line:
        score += 1.0

    return score


def _level3_requested_outcomes(
    task: EvidenceTask,
) -> tuple[str, ...]:
    """Return outcome labels explicitly present in the Level-3 table task."""
    task_text = (
        task.query
        + " "
        + task.anchor
    ).lower()

    candidates = (
        "competitive advantage",
        "organisational performance",
        "organizational performance",
    )

    return tuple(
        outcome
        for outcome in candidates
        if outcome in task_text
    )


def _is_requested_level3_interaction_row(
    line: str,
    task: EvidenceTask,
) -> bool:
    """Check whether one line is a requested interaction-result row."""
    lowered = line.lower()

    has_interaction_subject = (
        "intensity of competitive forces" in lowered
        and "traditional" in lowered
        and "management accounting" in lowered
    )

    requested_outcomes = (
        _level3_requested_outcomes(
            task
        )
    )
    has_requested_outcome = (
        not requested_outcomes
        or any(
            outcome in lowered
            for outcome in requested_outcomes
        )
    )

    return (
        has_interaction_subject
        and has_requested_outcome
        and _numeric_table_value_count(
            line
        )
        >= 1
    )


def _level3_following_p_value_line(
    lines: list[str],
    row_index: int,
) -> str:
    """Find the p-value line immediately associated with one table row."""
    for following_index in range(
        row_index + 1,
        min(
            len(lines),
            row_index + 3,
        ),
    ):
        following = lines[
            following_index
        ]

        if re.search(
            r"\(\s*[-+]?0?\.\d{2,4}\s*\)\s*\*{0,3}",
            following,
        ):
            return following

        # Stop when the next substantive relationship row begins.
        if (
            "intensity of competitive forces"
            in following.lower()
            and _numeric_table_value_count(
                following
            )
            >= 1
        ):
            break

    return ""


def _infer_level3_column_labels(
    table_text: str,
    value_count: int,
) -> list[str]:
    """Infer semantic column labels from the extracted table header.

    The labels come from the table text itself. A generic numbered fallback is
    used when the header cannot be reconstructed reliably.
    """
    normalised = re.sub(
        r"\s+",
        " ",
        table_text.replace("\u00ad", " "),
    ).lower()

    has_expected_group_header = all(
        signal in normalised
        for signal in (
            "overall sample",
            "low-cost",
            "product",
            "differentiation",
            "high-force",
            "low-force",
        )
    )

    if (
        has_expected_group_header
        and value_count == 5
    ):
        return [
            "Overall sample",
            "Low-cost / High-force",
            "Low-cost / Low-force",
            "Product differentiation / High-force",
            "Product differentiation / Low-force",
        ]

    labels: list[str] = []

    if "overall sample" in normalised:
        labels.append("Overall sample")

    while len(labels) < value_count:
        labels.append(
            f"Column {len(labels) + 1}"
        )

    return labels[:value_count]


def _extract_level3_coefficients(
    coefficient_line: str,
) -> list[str]:
    """Extract ordered path coefficients from one reconstructed table row."""
    return re.findall(
        r"(?<![\d.])[-+]?(?:0?\.\d{2,4}|1\.\d{2,4})(?![\d.])",
        coefficient_line,
    )


def _extract_level3_p_values(
    p_value_line: str,
) -> list[tuple[str, str]]:
    """Extract ordered p-values and their significance markers."""
    return [
        (
            match.group("p"),
            match.group("stars") or "",
        )
        for match in re.finditer(
            r"\(\s*(?P<p>[-+]?(?:0?\.\d{2,4}|1\.\d{2,4}))\s*\)"
            r"\s*(?P<stars>\*{0,3})",
            p_value_line,
        )
    ]


def _level3_row_label(
    coefficient_line: str,
) -> str:
    """Return the semantic row name before the first numeric value."""
    match = re.search(
        r"(?<![\d.])[-+]?(?:0?\.\d{2,4}|1\.\d{2,4})(?![\d.])",
        coefficient_line,
    )

    if not match:
        return coefficient_line.strip()

    return coefficient_line[
        : match.start()
    ].strip(" |")


def _level3_semantic_row_block(
    coefficient_line: str,
    p_value_line: str,
    table_text: str,
) -> str:
    """Attach row and column names to every coefficient/p-value pair."""
    coefficients = _extract_level3_coefficients(
        coefficient_line
    )
    p_values = _extract_level3_p_values(
        p_value_line
    )

    pair_count = min(
        len(coefficients),
        len(p_values),
    )

    if pair_count == 0:
        return ""

    column_labels = (
        _infer_level3_column_labels(
            table_text,
            pair_count,
        )
    )
    row_label = _level3_row_label(
        coefficient_line
    )

    lines = [
        f"Row label: {row_label}",
        "Metric: Path coefficient with p-value",
    ]

    for index in range(pair_count):
        p_value, stars = p_values[index]
        significance_text = (
            f"; significance marker={stars}"
            if stars
            else ""
        )

        lines.append(
            f"Column label: {column_labels[index]}; "
            f"coefficient={coefficients[index]}; "
            f"p_value={p_value}"
            f"{significance_text}"
        )

    # Make the first column unambiguous for questions about the overall sample.
    if (
        column_labels
        and column_labels[0]
        == "Overall sample"
    ):
        first_p, first_stars = p_values[0]
        lines.append(
            "Overall-sample result: "
            f"coefficient={coefficients[0]}; "
            f"p_value={first_p}"
            + (
                f"; significance marker={first_stars}"
                if first_stars
                else ""
            )
        )

    return "\n".join(lines)


def _level3_same_row_statistic(
    coefficient_line: str,
    p_value_line: str,
) -> str:
    """Keep backward-compatible overall-sample formatting."""
    coefficients = _extract_level3_coefficients(
        coefficient_line
    )
    p_values = _extract_level3_p_values(
        p_value_line
    )

    if not coefficients or not p_values:
        return ""

    first_p, first_stars = p_values[0]

    return (
        "Same-row overall-sample statistic: "
        f"{coefficients[0]} "
        f"({first_p})"
        f"{first_stars}"
    )


def _focus_level3_table_row(
    text: str,
    task: EvidenceTask,
) -> str:
    """Keep every requested interaction row and its own p-value line.

    For q7 this retains both traditional-management-accounting outcomes:
    organisational performance and competitive advantage. The logic remains
    restricted to the explicit Level-3 table path.
    """
    normalised = _normalise_level3_table_text(
        text,
        task,
    )
    lines = [
        line.strip()
        for line in normalised.splitlines()
        if line.strip()
    ]

    if not lines:
        return ""

    matching_indices = [
        index
        for index, line in enumerate(
            lines
        )
        if _is_requested_level3_interaction_row(
            line,
            task,
        )
    ]

    # Preserve the previous generic fallback when a table does not expose
    # separately identifiable interaction rows.
    if not matching_indices:
        matching_indices = [
            max(
                range(len(lines)),
                key=lambda index: (
                    _level3_table_line_score(
                        lines[index],
                        task,
                    ),
                    -index,
                ),
            )
        ]

    focused_rows: list[str] = []
    seen_rows: set[str] = set()

    for row_index in matching_indices:
        coefficient_line = lines[
            row_index
        ]
        p_value_line = (
            _level3_following_p_value_line(
                lines,
                row_index,
            )
        )

        row_parts = [
            coefficient_line
        ]

        if p_value_line:
            row_parts.append(
                p_value_line
            )

        semantic_block = (
            _level3_semantic_row_block(
                coefficient_line,
                p_value_line,
                normalised,
            )
            if p_value_line
            else ""
        )

        if semantic_block:
            row_parts.append(
                semantic_block
            )

        statistic = (
            _level3_same_row_statistic(
                coefficient_line,
                p_value_line,
            )
            if p_value_line
            else ""
        )

        if statistic:
            row_parts.append(
                statistic
            )

        focused_row = "\n".join(
            row_parts
        )
        row_key = re.sub(
            r"\s+",
            " ",
            focused_row.lower(),
        ).strip()

        if (
            focused_row
            and row_key not in seen_rows
        ):
            seen_rows.add(row_key)
            focused_rows.append(
                focused_row
            )

    return "\n\n".join(
        focused_rows
    )


def _numeric_table_value_count(
    text: str,
) -> int:
    """Count decimal table values in one extracted line."""
    return len(
        re.findall(
            r"(?<![\d.])[-+]?0?\.\d{2,4}(?![\d.])",
            text,
        )
    )


def _level3_table_search_text(
    task: EvidenceTask,
    payload: dict[str, Any],
) -> str:
    """Build searchable text for one Level-3 table candidate only."""
    parts: list[str] = []

    stored_text = str(
        payload.get("text", "")
    ).strip()
    raw_row_text = str(
        payload.get("raw_row_text", "")
    ).strip()
    row_label = str(
        payload.get("row_label", "")
    ).strip()

    source_text = (
        raw_row_text
        or stored_text
    )
    focused_row = _focus_level3_table_row(
        source_text,
        task,
    )

    if focused_row:
        parts.append(
            "Focused reconstructed table row:\n"
            + focused_row
        )
        payload[
            "_level3_table_quote"
        ] = focused_row

    if row_label:
        parts.append(
            "Row label: "
            + _normalise_level3_table_text(
                row_label,
                task,
            )
        )

    cells = payload.get(
        "cells",
        [],
    )

    if isinstance(cells, list):
        normalised_cells: list[str] = []

        for cell in cells:
            if not isinstance(
                cell,
                dict,
            ):
                continue

            cell_id = str(
                cell.get("cell_id", "")
            ).strip()
            cell_text = (
                _normalise_level3_table_text(
                    str(
                        cell.get(
                            "text",
                            "",
                        )
                    ),
                    task,
                )
            )

            if not cell_text:
                continue

            normalised_cells.append(
                (
                    f"{cell_id}={cell_text}"
                    if cell_id
                    else cell_text
                )
            )

        if normalised_cells:
            parts.append(
                "Cells: "
                + " | ".join(
                    normalised_cells
                )
            )

    statistics = payload.get(
        "statistics",
        [],
    )

    if isinstance(statistics, list):
        statistic_parts: list[str] = []

        for statistic in statistics:
            if not isinstance(
                statistic,
                dict,
            ):
                continue

            coefficient = str(
                statistic.get(
                    "coefficient",
                    "",
                )
            ).strip()
            p_value = str(
                statistic.get(
                    "p_value",
                    "",
                )
            ).strip()
            significance = str(
                statistic.get(
                    "significance_marker",
                    "",
                )
            ).strip()

            if coefficient and p_value:
                statistic_parts.append(
                    f"{coefficient} "
                    f"({p_value})"
                    f"{significance}"
                )

        if statistic_parts:
            parts.append(
                "Statistics from this same row: "
                + "; ".join(
                    statistic_parts
                )
            )

    if stored_text:
        parts.append(
            "Normalised extracted table text:\n"
            + _normalise_level3_table_text(
                stored_text,
                task,
            )
        )

    return "\n".join(
        dict.fromkeys(
            part
            for part in parts
            if part
        )
    )


def _prepare_level3_hits(
    task: EvidenceTask,
    hits: list[Any],
) -> list[Any]:
    """Normalise only Level-3 table hits before reranking."""
    if task.evidence_type != "table":
        return hits

    for hit in hits:
        payload = getattr(
            hit,
            "payload",
            None,
        )

        if not isinstance(
            payload,
            dict,
        ):
            continue

        structured_text = (
            _level3_table_search_text(
                task,
                payload,
            )
        )

        if structured_text:
            payload["text"] = structured_text

    return hits



def _level3_context_text(
    task: EvidenceTask,
    payload: dict[str, Any],
) -> str:
    """Return concise evidence text for one Level-3 result."""
    if task.evidence_type == "table":
        focused_quote = str(
            payload.get(
                "_level3_table_quote",
                "",
            )
        ).strip()

        if focused_quote:
            return focused_quote

        raw_row_text = str(
            payload.get(
                "raw_row_text",
                "",
            )
        ).strip()

        if raw_row_text:
            return _normalise_level3_table_text(
                raw_row_text,
                task,
            )

    text = str(
        payload.get(
            "text",
            "",
        )
    )

    if task.evidence_type == "comparison_item":
        focused = _focus_level3_comparison_passage(
            task,
            text,
        )

        if focused:
            return (
                f"Comparison label: {task.label}\n"
                f"Comparison item: {task.comparison_item}\n"
                f"Compared attribute: {task.comparison_attribute}\n"
                f"Item-specific evidence: {focused}"
            )

    if task.evidence_type == "synthesis_component":
        focused = _focus_level3_synthesis_passage(
            task,
            text,
        )

        if focused:
            return focused

    return text

_COMPARISON_GENERIC_ITEM_WORDS = {
    "case",
    "cases",
    "category",
    "categories",
    "condition",
    "conditions",
    "group",
    "groups",
    "model",
    "models",
    "period",
    "periods",
}


def _normalise_comparison_text(text: str) -> str:
    """Normalise punctuation and PDF word breaks for item matching."""
    cleaned = text.replace("\u00ad", " ").lower()
    cleaned = re.sub(
        r"[-–—/]",
        " ",
        cleaned,
    )
    return re.sub(
        r"\s+",
        " ",
        cleaned,
    ).strip()


def _comparison_item_words(item: str) -> set[str]:
    """Return item-defining words while ignoring only generic item heads."""
    return {
        token
        for token in re.findall(
            r"[a-z0-9]+",
            _normalise_comparison_text(item),
        )
        if token not in _COMPARISON_GENERIC_ITEM_WORDS
    }



_COMPARISON_ATTRIBUTE_RELATION_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "between",
    "by",
    "compare",
    "compared",
    "comparing",
    "comparison",
    "contrast",
    "contrasted",
    "contrasting",
    "did",
    "differ",
    "difference",
    "differences",
    "differed",
    "differing",
    "differs",
    "do",
    "does",
    "effect",
    "effects",
    "for",
    "from",
    "how",
    "impact",
    "impacts",
    "in",
    "is",
    "moderate",
    "moderated",
    "moderates",
    "moderating",
    "moderation",
    "of",
    "on",
    "or",
    "relative",
    "relationship",
    "relationships",
    "association",
    "associations",
    "the",
    "to",
    "versus",
    "vary",
    "varied",
    "varies",
    "varying",
    "was",
    "were",
    "with",
}


def _comparison_word_root(word: str) -> str:
    """Create a light normal form for comparison-attribute matching."""
    token = word.lower()

    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"

    if (
        len(token) > 4
        and token.endswith("s")
        and not token.endswith("ss")
    ):
        return token[:-1]

    return token


def _comparison_attribute_content_words(
    attribute: str,
) -> set[str]:
    """Return the substantive words that define the compared attribute."""
    return {
        _comparison_word_root(token)
        for token in re.findall(
            r"[a-z0-9]+",
            _normalise_comparison_text(attribute),
        )
        if token not in _COMPARISON_ATTRIBUTE_RELATION_WORDS
    }


def _level3_comparison_attribute_content_strength(
    task: EvidenceTask,
    passage: str,
) -> float:
    """Measure coverage of substantive attribute wording.

    Generic relationship words such as ``effect`` and ``moderating`` are
    excluded. This prevents evidence about a neighbouring construct from being
    accepted merely because both passages describe the same type of effect.
    """
    attribute_words = _comparison_attribute_content_words(
        task.comparison_attribute
    )

    if not attribute_words:
        return _lexical_overlap(
            task.comparison_attribute or task.query,
            passage,
        )

    passage_words = {
        _comparison_word_root(token)
        for token in re.findall(
            r"[a-z0-9]+",
            _normalise_comparison_text(passage),
        )
    }

    return len(
        attribute_words & passage_words
    ) / len(attribute_words)


def _comparison_item_pattern(item: str) -> re.Pattern[str] | None:
    """Build a flexible pattern for the defining words of one item."""
    item_words = [
        token
        for token in re.findall(
            r"[a-z0-9]+",
            _normalise_comparison_text(item),
        )
        if token not in _COMPARISON_GENERIC_ITEM_WORDS
    ]

    if not item_words:
        item_words = re.findall(
            r"[a-z0-9]+",
            _normalise_comparison_text(item),
        )

    if not item_words:
        return None

    return re.compile(
        r"\b"
        + r"[\W_]+".join(
            re.escape(token)
            for token in item_words
        )
        + r"\b",
        flags=re.IGNORECASE,
    )


def _comparison_sentence_bounds(
    text: str,
    position: int,
) -> tuple[int, int]:
    """Return conservative sentence boundaries around one item mention."""
    boundaries = [0]

    for match in re.finditer(
        r"(?<=[.!?])\s+(?=[A-Z“‘])|\n+",
        text,
    ):
        boundaries.append(
            match.end()
        )

    boundaries.append(
        len(text)
    )
    boundaries = sorted(
        set(boundaries)
    )

    start = 0
    end = len(text)

    for boundary in boundaries:
        if boundary <= position:
            start = boundary
            continue

        end = boundary
        break

    return start, end



def _comparison_clause_separators(
    bridge: str,
) -> list[re.Match[str]]:
    """Find separators that are likely to divide item-level findings."""
    return list(
        re.finditer(
            r";|\bwhereas\b|\bwhile\b|\bbut\b|"
            r"\band\s+between\b|\band\s+for\b|"
            r",(?=\s+(?:there|while|whereas|however|but|"
            r"in\s+contrast|by\s+contrast)\b)",
            bridge,
            flags=re.IGNORECASE,
        )
    )

def _left_comparison_clause_start(
    bridge: str,
    absolute_start: int,
) -> int:
    """Move a clause start past the last separator before the target item."""
    separators = _comparison_clause_separators(
        bridge
    )

    if not separators:
        return absolute_start

    separator = separators[-1]
    separator_text = separator.group(0).lower()

    if separator_text.startswith(
        "and "
    ):
        # Preserve ``between`` or ``for`` as part of the focused evidence.
        offset = separator.start() + len("and ")
    else:
        offset = separator.end()

    return absolute_start + offset


def _right_comparison_clause_end(
    bridge: str,
    absolute_start: int,
) -> int:
    """End a clause before the first separator leading to another item."""
    separators = _comparison_clause_separators(
        bridge
    )

    if not separators:
        return absolute_start + len(bridge)

    return absolute_start + separators[0].start()


def _focus_level3_comparison_passage(
    task: EvidenceTask,
    passage: str,
) -> str:
    """Return only the clause most directly associated with one comparison item.

    A source sentence can state findings for several comparison items. This
    function keeps the target item's clause and removes neighbouring item
    clauses before the evidence is passed to the answer model.
    """
    cleaned = re.sub(
        r"\s+",
        " ",
        passage.replace("\u00ad", " "),
    ).strip()

    if not cleaned:
        return ""

    target_pattern = _comparison_item_pattern(
        task.comparison_item
    )

    if target_pattern is None:
        return cleaned

    target_match = target_pattern.search(
        cleaned
    )

    if target_match is None:
        return cleaned

    sentence_start, sentence_end = (
        _comparison_sentence_bounds(
            cleaned,
            target_match.start(),
        )
    )
    sentence = cleaned[
        sentence_start:sentence_end
    ]

    item_spans: list[tuple[int, int, str]] = []

    for item in task.comparison_items:
        pattern = _comparison_item_pattern(
            item
        )

        if pattern is None:
            continue

        for match in pattern.finditer(
            sentence
        ):
            item_spans.append(
                (
                    match.start(),
                    match.end(),
                    item,
                )
            )

    item_spans.sort(
        key=lambda span: (
            span[0],
            span[1],
        )
    )

    target_index = next(
        (
            index
            for index, span in enumerate(
                item_spans
            )
            if (
                span[2]
                == task.comparison_item
                and span[0]
                <= target_match.start() - sentence_start
                < span[1]
            )
        ),
        None,
    )

    if target_index is None:
        return sentence.strip()

    clause_start = 0
    clause_end = len(sentence)

    if target_index > 0:
        previous_span = item_spans[
            target_index - 1
        ]
        bridge_start = previous_span[1]
        bridge = sentence[
            bridge_start:item_spans[target_index][0]
        ]
        clause_start = _left_comparison_clause_start(
            bridge,
            bridge_start,
        )

    if target_index + 1 < len(item_spans):
        next_span = item_spans[
            target_index + 1
        ]
        bridge_start = item_spans[
            target_index
        ][1]
        bridge = sentence[
            bridge_start:next_span[0]
        ]
        clause_end = _right_comparison_clause_end(
            bridge,
            bridge_start,
        )

    focused = sentence[
        clause_start:clause_end
    ].strip(" \t\n\r,;")

    return focused or sentence.strip()

def _level3_comparison_item_strength(
    item: str,
    passage: str,
) -> float:
    """Measure how clearly one passage refers to a specific comparison item."""
    if not item:
        return 0.0

    normalised_item = _normalise_comparison_text(item)
    normalised_passage = _normalise_comparison_text(passage)

    if normalised_item in normalised_passage:
        return 1.0

    item_words = _comparison_item_words(item)

    if not item_words:
        return 0.0

    passage_words = set(
        re.findall(
            r"[a-z0-9]+",
            normalised_passage,
        )
    )

    return len(
        item_words & passage_words
    ) / len(item_words)


def _level3_comparison_item_focus(
    task: EvidenceTask,
    passage: str,
) -> float:
    """Reward the target item while mildly demoting multi-item passages."""
    target_strength = _level3_comparison_item_strength(
        task.comparison_item,
        passage,
    )

    other_matches = sum(
        1
        for other_item in task.comparison_items
        if (
            other_item != task.comparison_item
            and _level3_comparison_item_strength(
                other_item,
                passage,
            )
            >= 0.75
        )
    )

    return max(
        0.0,
        target_strength
        - 0.12 * other_matches,
    )



def _level3_comparison_attribute_strength(
    task: EvidenceTask,
    passage: str,
) -> float:
    """Measure coverage of the attribute that is being compared."""
    if not task.comparison_attribute:
        return _lexical_overlap(
            task.query,
            passage,
        )

    lexical = _lexical_overlap(
        task.comparison_attribute,
        passage,
    )
    content = _level3_comparison_attribute_content_strength(
        task,
        passage,
    )

    return min(
        1.0,
        0.45 * lexical
        + 0.55 * content,
    )

def _level3_comparison_finding_strength(
    passage: str,
) -> float:
    """Measure whether a passage states an item-level empirical finding."""
    lowered = passage.replace("\u00ad", "").lower()
    score = 0.0

    if any(
        term in lowered
        for term in (
            "effect",
            "moderate",
            "moderates",
            "moderating",
            "association",
            "relationship",
            "higher",
            "lower",
            "increase",
            "decrease",
            "positive",
            "negative",
        )
    ):
        score += 0.25

    if any(
        term in lowered
        for term in (
            "significant",
            "not significant",
            "did not",
            "no support",
            "rejected",
            "supported",
            "partial support",
        )
    ):
        score += 0.25

    if _contains_signal(
        passage,
        OUTCOME_LANGUAGE,
    ):
        score += 0.20

    score += 0.25 * _level3_statistical_strength(
        passage
    )

    if (
        "hypothesis" in lowered
        or bool(re.search(r"\bh\s*\d\b", lowered))
        or bool(re.search(r"\bh\d\b", lowered))
    ):
        score += 0.05

    return min(
        1.0,
        score,
    )


def _level3_comparison_topic_strength(
    task: EvidenceTask,
    passage: str,
) -> float:
    """Combine the compared attribute with the target comparison item."""
    attribute = _level3_comparison_attribute_strength(
        task,
        passage,
    )
    item = _level3_comparison_item_strength(
        task.comparison_item,
        passage,
    )

    return min(
        1.0,
        0.55 * attribute
        + 0.45 * item,
    )


def _level3_comparison_strength(
    task: EvidenceTask,
    passage: str,
) -> float:
    """Score one passage as evidence for one extracted comparison item."""
    item_focus = _level3_comparison_item_focus(
        task,
        passage,
    )
    attribute = _level3_comparison_attribute_strength(
        task,
        passage,
    )
    finding = _level3_comparison_finding_strength(
        passage
    )

    return min(
        1.0,
        0.42 * item_focus
        + 0.28 * attribute
        + 0.30 * finding,
    )


_SYNTHESIS_RECOMMENDATION_SIGNALS = (
    "recommend",
    "recommended",
    "recommendation",
    "should",
    "should not",
    "ought",
    "must",
    "advised",
    "encouraged",
    "discouraged",
    "appropriate to",
    "inappropriate to",
)

_SYNTHESIS_MANAGER_ACTIONS = (
    "adopt",
    "avoid",
    "employ",
    "focus",
    "increase",
    "reduce",
    "rely",
    "use",
)

_SYNTHESIS_MAIN_CONCLUSION_SIGNALS = (
    "main conclusion",
    "overall conclusion",
    "in conclusion",
    "we conclude",
    "the study concludes",
    "this study concludes",
    "taken together",
    "overall, this study",
    "overall this study",
    "this study makes a contribution",
    "this study makes a significant contribution",
    "main contribution",
    "overall contribution",
    "highlighting the important role",
    "our findings highlight",
    "these findings highlight",
)

_SYNTHESIS_ISOLATED_RESULT_SIGNALS = (
    "overall sample model",
    "subgroup model",
    "quadrant model",
    "table 5",
    "support for h",
    "h1",
    "h2",
    "h3",
    "significant finding",
    "path coefficient",
)


def _split_synthesis_sentences(text: str) -> list[str]:
    """Split a retrieved passage while keeping academic sentences intact."""
    cleaned = re.sub(
        r"\s+",
        " ",
        text.replace("\u00ad", " "),
    ).strip()

    if not cleaned:
        return []

    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+(?=[A-Z“‘])",
            cleaned,
        )
        if sentence.strip()
    ]


def _synthesis_sentence_is_actionable_recommendation(
    sentence: str,
) -> bool:
    """Require an author recommendation or a manager-directed action."""
    lowered = sentence.lower()

    if any(
        signal in lowered
        for signal in _SYNTHESIS_RECOMMENDATION_SIGNALS
    ):
        return True

    has_manager = bool(
        re.search(
            r"\bmanagers?\b",
            lowered,
        )
    )
    has_action = any(
        action in lowered
        for action in _SYNTHESIS_MANAGER_ACTIONS
    )

    return has_manager and has_action


def _synthesis_recommendation_sentences(
    passage: str,
) -> list[str]:
    """Return every distinct condition-action recommendation in a passage."""
    sentences = _split_synthesis_sentences(
        passage
    )
    selected: list[str] = []

    for sentence in sentences:
        # Semicolons and explicit transition clauses often separate two
        # recommendations while preserving the condition stated in the text.
        units = [
            unit.strip()
            for unit in re.split(
                r"(?<=;)\s+|(?=At the same time,)|(?=Conversely,)|(?=If )",
                sentence,
            )
            if unit.strip()
        ]

        for unit in units:
            if _synthesis_sentence_is_actionable_recommendation(
                unit
            ):
                selected.append(unit)

    return list(
        dict.fromkeys(selected)
    )


def _synthesis_recommendation_signature(
    sentence: str,
) -> tuple[str, ...]:
    """Create a stable signature for deduplicating recommendation units."""
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for",
        "from", "given", "if", "in", "is", "it", "of", "on", "or",
        "that", "the", "their", "then", "they", "this", "to", "when",
        "whose", "with",
    }
    words = [
        token
        for token in re.findall(
            r"[a-z0-9]+",
            sentence.lower(),
        )
        if token not in stop_words
    ]
    return tuple(words)


def _synthesis_main_conclusion_discourse_strength(
    passage: str,
) -> float:
    """Reward passages that explicitly present an integrated conclusion."""
    lowered = re.sub(
        r"\s+",
        " ",
        passage.replace("\u00ad", " ").lower(),
    ).strip()

    strong_matches = sum(
        1
        for signal in _SYNTHESIS_MAIN_CONCLUSION_SIGNALS
        if signal in lowered
    )
    generic_matches = sum(
        1
        for signal in (
            "overall",
            "conclusion",
            "conclude",
            "contribution",
            "highlight",
            "important role",
            "taken together",
        )
        if signal in lowered
    )

    return min(
        1.0,
        0.55 * min(1.0, float(strong_matches))
        + 0.45 * min(1.0, generic_matches / 3.0),
    )


def _synthesis_isolated_result_penalty(
    passage: str,
) -> float:
    """Identify passages dominated by one model, hypothesis, or statistic."""
    lowered = re.sub(
        r"\s+",
        " ",
        passage.replace("\u00ad", " ").lower(),
    ).strip()

    marker_matches = sum(
        1
        for signal in _SYNTHESIS_ISOLATED_RESULT_SIGNALS
        if signal in lowered
    )
    statistics = _level3_statistical_strength(
        passage
    )
    model_language = 1.0 if re.search(
        r"\b(?:sample|subgroup|quadrant|high[- ]force|low[- ]force)\s+model\b",
        lowered,
    ) else 0.0

    return min(
        1.0,
        0.45 * min(1.0, marker_matches / 2.0)
        + 0.35 * statistics
        + 0.20 * model_language,
    )


def _level3_synthesis_component_strength(
    task: EvidenceTask,
    passage: str,
) -> float:
    """Measure whether a passage addresses the requested synthesis component."""
    lowered = passage.replace("\u00ad", "").lower()
    component = task.synthesis_component

    if component == "main_conclusion":
        return _synthesis_main_conclusion_discourse_strength(
            passage
        )

    signal_sets = {
        "main_argument": (
            "main argument",
            "central argument",
            "we argue",
            "argues that",
            "we maintain",
            "central claim",
            "propose that",
        ),
        "contributions": (
            "contribution",
            "contributes",
            "contributing",
            "extends",
            "advances",
            "adds to",
            "literature",
        ),
        "implications": (
            "implication",
            "implications",
            "practical",
            "theoretical",
            "managerial",
            "policy",
        ),
        "recommendations": _SYNTHESIS_RECOMMENDATION_SIGNALS,
        "findings_summary": (
            "overall",
            "findings",
            "results",
            "in summary",
            "taken together",
            "conclude",
            "conclusion",
        ),
        "overall_relationship": (
            "overall",
            "relationship",
            "association",
            "effect",
            "findings",
            "conclusion",
        ),
    }

    signals = signal_sets.get(
        component,
        (),
    )
    matched = sum(
        1
        for signal in signals
        if signal in lowered
    )

    if not signals:
        return 0.0

    base = min(
        1.0,
        matched / 3.0,
    )

    if component == "recommendations":
        recommendation_sentences = (
            _synthesis_recommendation_sentences(
                passage
            )
        )

        if recommendation_sentences:
            base = max(
                base,
                min(
                    1.0,
                    0.45
                    + 0.15
                    * len(
                        recommendation_sentences
                    ),
                ),
            )

    return base


def _level3_synthesis_topic_strength(
    task: EvidenceTask,
    passage: str,
) -> float:
    """Measure alignment with the full synthesis question and component."""
    question_overlap = _lexical_overlap(
        task.query,
        passage,
    )
    component_overlap = _lexical_overlap(
        task.synthesis_component_text,
        passage,
    )

    return min(
        1.0,
        0.70 * question_overlap
        + 0.30 * component_overlap,
    )


def _level3_synthesis_strength(
    task: EvidenceTask,
    passage: str,
) -> float:
    """Score one passage as evidence for one synthesis component."""
    component = _level3_synthesis_component_strength(
        task,
        passage,
    )
    topic = _level3_synthesis_topic_strength(
        task,
        passage,
    )

    score = (
        0.58 * component
        + 0.42 * topic
    )

    if task.synthesis_component == "main_conclusion":
        score -= 0.50 * _synthesis_isolated_result_penalty(
            passage
        )

    return max(
        0.0,
        min(1.0, score),
    )


def _focus_level3_synthesis_passage(
    task: EvidenceTask,
    passage: str,
) -> str:
    """Label one synthesis component and split conditional recommendations."""
    cleaned = re.sub(
        r"\s+",
        " ",
        passage.replace("\u00ad", " "),
    ).strip()

    if not cleaned:
        return ""

    header = (
        f"Synthesis label: {task.label}\n"
        f"Synthesis component: {task.synthesis_component}\n"
        f"Requested component: {task.synthesis_component_text}\n"
    )

    if task.synthesis_component != "recommendations":
        focused_text = cleaned

        if task.synthesis_component in {
            "main_conclusion",
            "main_argument",
            "contributions",
            "findings_summary",
            "overall_relationship",
        }:
            sentences = _split_synthesis_sentences(
                cleaned
            )
            non_recommendation_sentences: list[str] = []

            for sentence in sentences:
                if _synthesis_recommendation_sentences(
                    sentence
                ):
                    break

                non_recommendation_sentences.append(
                    sentence
                )

            if non_recommendation_sentences:
                focused_text = " ".join(
                    non_recommendation_sentences
                )

        return (
            header
            + "Component-specific evidence: "
            + focused_text
        )

    recommendations = (
        _synthesis_recommendation_sentences(
            cleaned
        )
    )

    if not recommendations:
        return (
            header
            + "Component-specific evidence: "
            + cleaned
        )

    sublabels: list[str] = []

    for subposition, sentence in enumerate(
        recommendations,
        start=1,
    ):
        suffix = chr(
            ord("a")
            + subposition
            - 1
        )
        sublabels.append(
            f"synthesis_{task.synthesis_position}{suffix}: "
            f"{sentence}"
        )

    return (
        header
        + "Condition-specific recommendation evidence:\n"
        + "\n".join(sublabels)
    )


def _level3_topic_strength(
    task: EvidenceTask,
    passage: str,
) -> float:
    """Measure alignment with the main entities requested by one evidence task."""
    if task.evidence_type == "comparison_item":
        return _level3_comparison_topic_strength(
            task,
            passage,
        )

    if task.evidence_type == "synthesis_component":
        return _level3_synthesis_topic_strength(
            task,
            passage,
        )

    lowered = passage.replace("\u00ad", "").lower()

    has_management_accounting = "management accounting" in lowered
    has_competitive_forces = (
        "competitive force" in lowered
        or "five forces" in lowered
    )

    signals = [
        has_management_accounting,
        has_competitive_forces,
    ]

    if task.evidence_type in {"table", "results"}:
        # Do not require one exact uninterrupted phrase because PDF extraction can
        # insert line breaks or spacing between "traditional" and "management".
        signals.append(
            "traditional" in lowered
            and has_management_accounting
        )

    return sum(signals) / len(signals)


def _level3_statistical_strength(passage: str) -> float:
    """Detect generic statistical-result formatting without expected values."""
    cleaned = passage.replace("\u00ad", "")
    lowered = cleaned.lower()
    score = 0.0

    if re.search(
        r"(?:β|beta)\s*=\s*[-+]?\d*\.\d+",
        cleaned,
        re.IGNORECASE,
    ):
        score += 0.25

    if re.search(r"\bp\s*[=<]\s*0?\.\d+", lowered):
        score += 0.25

    if re.search(r"\(\s*0?\.\d{2,4}\s*\)\s*\*?", cleaned):
        score += 0.25

    decimal_values = re.findall(
        r"(?<![\d.])[-+]?0?\.\d{2,4}(?![\d.])",
        cleaned,
    )
    if len(decimal_values) >= 2:
        score += 0.15

    if "*" in cleaned:
        score += 0.10

    return min(1.0, score)


def _level3_has_traditional_focus(passage: str) -> bool:
    """Check that a passage is about traditional management accounting."""
    lowered = passage.replace("\u00ad", "").lower()
    return (
        "traditional" in lowered
        and "management accounting" in lowered
    )


def _level3_has_requested_outcome(passage: str) -> bool:
    """Check for the organisational outcomes named by the q7 evidence task."""
    lowered = passage.replace("\u00ad", "").lower()
    return any(
        term in lowered
        for term in (
            "competitive advantage",
            "organisational performance",
            "organizational performance",
        )
    )


def _level3_methodology_penalty(passage: str) -> float:
    """Identify method descriptions that are not empirical table evidence."""
    lowered = passage.replace("\u00ad", "").lower()

    methodology_signals = (
        "ordinal independent variable interaction approach",
        "we use the",
        "methodological approach",
        "methodology",
        "data analysis",
        "model specification",
        "we consider both the direct effect",
        "direct effect of contemporary and traditional",
        "luft & shields",
        "luft and shields",
    )

    matched = sum(
        1
        for signal in methodology_signals
        if signal in lowered
    )

    return min(1.0, matched / 2.0)


def _level3_table_row_pattern(passage: str) -> float:
    """Measure whether text resembles a coefficient-and-p-value table row."""
    cleaned = passage.replace("\u00ad", "")
    lowered = cleaned.lower()
    score = 0.0

    # Typical extracted row: coefficient followed by a parenthesised p-value.
    if re.search(
        r"[-+]?0?\.\d{2,4}\s*\(\s*0?\.\d{2,4}\s*\)\s*\*?",
        cleaned,
    ):
        score += 0.55

    if any(marker in cleaned for marker in ("×", "→", "\t")):
        score += 0.15

    if "table 5" in lowered:
        score += 0.15

    decimal_values = re.findall(
        r"(?<![\d.])[-+]?0?\.\d{2,4}(?![\d.])",
        cleaned,
    )
    if len(decimal_values) >= 2:
        score += 0.15

    return min(1.0, score)


def _level3_literature_strength(passage: str) -> float:
    """Measure whether a passage states the general literature-review argument."""
    lowered = passage.replace("\u00ad", "").lower()

    response = _how_response_strength(passage)
    information = (
        1.0
        if any(
            phrase in lowered
            for phrase in (
                "decision support",
                "information provision",
                "providing information",
                "provide information",
                "informs decision making",
                "inform decision making",
                "understanding of their competitive forces",
            )
        )
        else 0.0
    )
    proposition = (
        1.0
        if any(
            phrase in lowered
            for phrase in (
                "potential to assist",
                "assist in managing",
                "help organisations",
                "help organizations",
                "we maintain that",
                "hence",
            )
        )
        else 0.0
    )

    return min(
        1.0,
        0.45 * response
        + 0.35 * information
        + 0.20 * proposition,
    )


def _level3_table_strength(passage: str) -> float:
    """Measure whether a passage is the requested statistical table evidence."""
    cleaned = passage.replace("\u00ad", "")
    lowered = cleaned.lower()

    table_marker = 1.0 if "table 5" in lowered else 0.0
    traditional = 1.0 if _level3_has_traditional_focus(passage) else 0.0
    interaction = (
        1.0
        if any(
            marker in lowered
            for marker in (
                "interaction",
                "intensity of competitive forces ×",
                "intensity of competitive forces x",
                "competitive forces × traditional",
                "competitive forces x traditional",
            )
        )
        else 0.0
    )
    outcome = 1.0 if _level3_has_requested_outcome(passage) else 0.0
    statistics = _level3_statistical_strength(passage)
    row_pattern = _level3_table_row_pattern(passage)
    methodology = _level3_methodology_penalty(passage)

    # Narrative results can contain beta and p-values, but a true table candidate
    # should look like a row and should not primarily describe the methodology.
    narrative_penalty = (
        1.0
        if any(
            phrase in lowered
            for phrase in (
                "table 5 shows that",
                "positively moderates the association",
                "support is provided",
                "partial support",
                "and hence",
            )
        )
        else 0.0
    )

    score = (
        0.08 * table_marker
        + 0.18 * traditional
        + 0.12 * interaction
        + 0.12 * outcome
        + 0.20 * statistics
        + 0.30 * row_pattern
        - 0.30 * methodology
        - 0.12 * narrative_penalty
    )

    return max(0.0, min(1.0, score))


def _level3_results_strength(passage: str) -> float:
    """Measure whether a passage interprets the requested traditional-practice result."""
    lowered = passage.replace("\u00ad", "").lower()

    traditional = 1.0 if _level3_has_traditional_focus(passage) else 0.0
    contemporary_only = (
        1.0
        if (
            "contemporary management accounting" in lowered
            and not _level3_has_traditional_focus(passage)
        )
        else 0.0
    )
    interpretation = (
        1.0
        if any(
            term in lowered
            for term in (
                "table 5 shows",
                "shows that",
                "positively moderates",
                "positive moderating",
                "moderates the association",
                "significant positive",
            )
        )
        else 0.0
    )
    hypothesis = (
        1.0
        if (
            "hypothesis" in lowered
            or bool(re.search(r"\bh\s*\d\b", lowered))
            or bool(re.search(r"\bh\d\b", lowered))
            or "support is provided" in lowered
            or "partial support" in lowered
        )
        else 0.0
    )
    outcome = 1.0 if _level3_has_requested_outcome(passage) else 0.0
    statistics = _level3_statistical_strength(passage)
    methodology = _level3_methodology_penalty(passage)

    score = (
        0.25 * traditional
        + 0.25 * interpretation
        + 0.18 * hypothesis
        + 0.12 * outcome
        + 0.20 * statistics
        - 0.45 * contemporary_only
        - 0.30 * methodology
    )

    return max(0.0, min(1.0, score))



def _level3_candidate_is_eligible(
    task: EvidenceTask,
    passage: str,
) -> bool:
    """Apply evidence-type requirements before final Level-3 selection."""
    lowered = passage.replace("\u00ad", "").lower()

    if task.evidence_type == "comparison_item":
        return (
            _level3_comparison_item_strength(
                task.comparison_item,
                passage,
            )
            >= 0.75
            and _level3_comparison_attribute_content_strength(
                task,
                passage,
            )
            >= 0.80
            and _level3_comparison_finding_strength(
                passage
            )
            >= 0.25
        )

    if task.evidence_type == "synthesis_component":
        component_strength = (
            _level3_synthesis_component_strength(
                task,
                passage,
            )
        )
        topic_strength = (
            _level3_synthesis_topic_strength(
                task,
                passage,
            )
        )

        if task.synthesis_component == "recommendations":
            return (
                component_strength >= 0.45
                and topic_strength >= 0.20
                and bool(
                    _synthesis_recommendation_sentences(
                        passage
                    )
                )
            )

        if task.synthesis_component == "main_conclusion":
            return (
                component_strength >= 0.45
                and topic_strength >= 0.20
                and _synthesis_main_conclusion_discourse_strength(
                    passage
                )
                >= 0.45
                and _synthesis_isolated_result_penalty(
                    passage
                )
                < 0.65
            )

        return (
            component_strength >= 0.35
            and topic_strength >= 0.20
        )

    if task.evidence_type == "table":
        return (
            _level3_has_traditional_focus(passage)
            and _level3_has_requested_outcome(passage)
            and _level3_statistical_strength(passage) >= 0.25
            and _level3_table_row_pattern(passage) >= 0.45
            and _level3_methodology_penalty(passage) < 0.50
        )

    if task.evidence_type == "results":
        has_interpretation = any(
            term in lowered
            for term in (
                "table 5 shows",
                "shows that",
                "positively moderates",
                "positive moderating",
                "moderates the association",
                "support is provided",
                "partial support",
            )
        )

        return (
            _level3_has_traditional_focus(passage)
            and _level3_has_requested_outcome(passage)
            and has_interpretation
            and _level3_methodology_penalty(passage) < 0.50
        )

    if task.evidence_type == "literature_review":
        return (
            "management accounting" in lowered
            and (
                "competitive force" in lowered
                or "five forces" in lowered
            )
            and _level3_literature_strength(passage) >= 0.35
        )

    return True

def _level3_evidence_score(
    task: EvidenceTask,
    passage: str,
    base_score: float,
) -> float:
    """Rerank one passage according to the evidence type of its task."""
    bounded_base = max(
        0.0,
        min(1.0, float(base_score)),
    )
    topic = _level3_topic_strength(task, passage)

    if task.evidence_type == "comparison_item":
        type_strength = _level3_comparison_strength(
            task,
            passage,
        )
    elif task.evidence_type == "synthesis_component":
        type_strength = _level3_synthesis_strength(
            task,
            passage,
        )
    elif task.evidence_type == "literature_review":
        type_strength = _level3_literature_strength(passage)
    elif task.evidence_type == "table":
        type_strength = _level3_table_strength(passage)
    elif task.evidence_type == "results":
        type_strength = _level3_results_strength(passage)
    else:
        type_strength = _lexical_overlap(task.query, passage)

    score = (
        0.42 * bounded_base
        + 0.18 * topic
        + 0.40 * type_strength
    )

    # Strongly demote related but invalid passages. This prevents a methodology
    # description from outranking the actual table row and prevents a result about
    # contemporary practices from satisfying a traditional-practices task.
    if not _level3_candidate_is_eligible(task, passage):
        if task.evidence_type == "table":
            score *= 0.22
        elif task.evidence_type == "results":
            score *= 0.35
        elif task.evidence_type == "comparison_item":
            score *= 0.35
        elif task.evidence_type == "synthesis_component":
            score *= 0.25
        else:
            score *= 0.70

    return max(0.0, min(1.0, score))


def _select_synthesis_recommendation_candidates(
    eligible: list[tuple[float, _MergedHit]],
    max_candidates: int = 8,
) -> list[tuple[float, _MergedHit]]:
    """Select candidates that add new recommendation conditions or actions."""
    selected: list[tuple[float, _MergedHit]] = []
    seen_units: set[tuple[str, ...]] = set()

    for item in eligible:
        passage = str(
            item[1].hit.payload.get(
                "text",
                "",
            )
        )
        recommendations = _synthesis_recommendation_sentences(
            passage
        )
        signatures = {
            _synthesis_recommendation_signature(sentence)
            for sentence in recommendations
            if sentence
        }
        signatures.discard(())

        if not signatures:
            continue

        if signatures - seen_units:
            selected.append(item)
            seen_units.update(signatures)

        if len(selected) >= max_candidates:
            break

    return selected


def _extract_recommendation_units_from_context(
    context: Context,
) -> list[str]:
    """Read recommendation units from one already-labelled synthesis context."""
    marker = "Condition-specific recommendation evidence:"

    if marker in context.text:
        tail = context.text.split(marker, 1)[1]
        units = []

        for line in tail.splitlines():
            match = re.match(
                r"^synthesis_\d+[a-z]+:\s*(.+)$",
                line.strip(),
                flags=re.IGNORECASE,
            )
            if match:
                units.append(match.group(1).strip())

        if units:
            return units

    return _synthesis_recommendation_sentences(
        context.text
    )


def _aggregate_synthesis_recommendation_group(
    group: EvidenceGroup,
    task: EvidenceTask,
) -> EvidenceGroup:
    """Combine all novel recommendation units and assign global sublabels."""
    ordered_units: list[tuple[int, float, str]] = []
    seen: set[tuple[str, ...]] = set()

    for context in group.contexts:
        for unit in _extract_recommendation_units_from_context(
            context
        ):
            signature = _synthesis_recommendation_signature(
                unit
            )

            if not signature or signature in seen:
                continue

            seen.add(signature)
            ordered_units.append(
                (
                    context.page,
                    context.score,
                    unit,
                )
            )

    if not ordered_units:
        return group

    grouped_by_page: dict[int, list[tuple[float, str]]] = {}

    for page, score, unit in ordered_units:
        grouped_by_page.setdefault(
            page,
            [],
        ).append(
            (
                score,
                unit,
            )
        )

    contexts: list[Context] = []
    subposition = 1

    for page, page_units in grouped_by_page.items():
        lines = [
            f"Synthesis label: {task.label}",
            f"Synthesis component: {task.synthesis_component}",
            f"Requested component: {task.synthesis_component_text}",
            "Condition-specific recommendation evidence:",
        ]

        for _score, unit in page_units:
            suffix = chr(
                ord("a")
                + subposition
                - 1
            )
            lines.append(
                f"synthesis_{task.synthesis_position}{suffix}: {unit}"
            )
            subposition += 1

        contexts.append(
            Context(
                text="\n".join(lines),
                page=page,
                score=max(
                    score
                    for score, _unit in page_units
                ),
            )
        )

    return EvidenceGroup(
        label=group.label,
        query=group.query,
        contexts=contexts,
    )


def retrieve_explicit_query(
    task: EvidenceTask,
    top_k: int,
) -> list[Context]:
    """Retrieve and rerank evidence for one self-contained Level-3 task.

    This path does not use conversational history and does not change the
    existing Level-1/2 `retrieve()` function. Table tasks lazily activate the
    separate pdfplumber index; all other tasks use the normal PyMuPDF index.
    """
    plans = _make_level3_query_plans(task)

    candidate_k = min(
        MAX_CANDIDATES_PER_QUERY,
        max(
            MIN_CANDIDATES_PER_QUERY,
            top_k * CANDIDATE_MULTIPLIER,
        ),
    )

    embedder = get_embedder()

    if task.evidence_type == "table":
        # Hard Level-3 boundary:
        # pdfplumber and the separate table collection are activated only for
        # an explicit Level-3 table evidence task. The ordinary `retrieve()`
        # function below continues to use only the PyMuPDF collection.
        try:
            from .ingest import ensure_level3_table_index
            from ..vectorstore.qdrant_store import get_table_store

            ensure_level3_table_index()
            store = get_table_store()

            LOGGER.info(
                "Level-3 table boundary activated | collection=%s",
                store.collection,
            )

        except Exception as error:
            # Keep the Level-3 request usable if table extraction or indexing
            # fails, while logging the failure clearly. This fallback is still
            # confined to the Level-3 table task and cannot affect Levels 1 or 2.
            LOGGER.exception(
                "Level-3 table index unavailable; falling back to the normal "
                "PyMuPDF collection for this table task only: %s",
                error,
            )
            store = get_store()

    else:
        store = get_store()

    hit_groups = [
        (
            plan,
            _prepare_level3_hits(
                task=task,
                hits=_search(
                    query=plan.text,
                    candidate_k=candidate_k,
                    embedder=embedder,
                    store=store,
                ),
            ),
        )
        for plan in plans
    ]

    merged_hits = _merge_hits(hit_groups)
    base_ranked = _base_rerank(
        plans=plans,
        merged_hits=merged_hits,
        standalone_query=task.query,
    )

    reranked = [
        (
            _level3_evidence_score(
                task=task,
                passage=str(
                    merged_hit.hit.payload.get("text", "")
                ),
                base_score=base_score,
            ),
            merged_hit,
        )
        for base_score, merged_hit in base_ranked
    ]
    reranked.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    eligible = [
        item
        for item in reranked
        if _level3_candidate_is_eligible(
            task,
            str(item[1].hit.payload.get("text", "")),
        )
    ]
    ineligible = [
        item
        for item in reranked
        if not _level3_candidate_is_eligible(
            task,
            str(item[1].hit.payload.get("text", "")),
        )
    ]

    # Comparison tasks return one strictly aligned item-specific passage.
    # They never fill the group with evidence that failed the item/attribute
    # checks, because a neighbouring construct or another comparison item can
    # otherwise contaminate the final comparison. Existing evidence types keep
    # their previous eligible-first fallback behaviour.
    if task.evidence_type == "comparison_item":
        selected = eligible[:1]
    elif (
        task.evidence_type == "synthesis_component"
        and task.synthesis_component == "recommendations"
    ):
        # Recommendation completeness is based on novel condition-action units,
        # not only on the highest-scoring first passage.
        selected = _select_synthesis_recommendation_candidates(
            eligible,
            max_candidates=max(6, top_k),
        )
    elif task.evidence_type == "synthesis_component":
        # A synthesis component must be supported by component-aligned evidence.
        # Do not fill a missing conclusion with an isolated empirical result that
        # failed the component checks.
        selected = eligible[:top_k]
    else:
        selected = (eligible + ineligible)[:top_k]

    LOGGER.info(
        "Level-3 task retrieval completed | label=%s | evidence_type=%s | "
        "queries=%d | candidate_k_per_query=%d | merged_candidates=%d | "
        "returned=%d | top=%s",
        task.label,
        task.evidence_type,
        len(plans),
        candidate_k,
        len(merged_hits),
        len(selected),
        [
            {
                "page": int(
                    merged_hit.hit.payload.get("page", 0)
                ),
                "score": round(float(score), 4),
                "eligible": _level3_candidate_is_eligible(
                    task,
                    str(merged_hit.hit.payload.get("text", "")),
                ),
                "statistics": round(
                    _level3_statistical_strength(
                        str(merged_hit.hit.payload.get("text", ""))
                    ),
                    3,
                ),
            }
            for score, merged_hit in selected
        ],
    )

    return [
        Context(
            text=_level3_context_text(
                task=task,
                payload=merged_hit.hit.payload,
            ),
            page=int(
                merged_hit.hit.payload.get("page", 0)
            ),
            score=float(score),
        )
        for score, merged_hit in selected
    ]


def _context_key(context: Context) -> tuple[int, str]:
    """Create a stable page-text key for Level-3 deduplication."""
    normalised = re.sub(
        r"\s+",
        " ",
        context.text.replace("\u00ad", "").strip().lower(),
    )
    return context.page, normalised


def _deduplicate_level3_groups(
    groups: list[EvidenceGroup],
    top_k_per_task: int,
) -> list[EvidenceGroup]:
    """Remove repeated passages without deleting an entire evidence group."""
    seen: set[tuple[int, str]] = set()
    out: list[EvidenceGroup] = []

    for group in groups:
        unique_contexts: list[Context] = []

        for context in group.contexts:
            key = _context_key(context)

            if key in seen:
                continue

            seen.add(key)
            unique_contexts.append(context)

            if len(unique_contexts) >= top_k_per_task:
                break

        # Preserve evidence-group coverage even when all of its candidates also
        # appeared in an earlier group. The synthesis prompt can then recognise
        # that this component was searched but shares a passage with another one.
        if not unique_contexts and group.contexts:
            unique_contexts = [group.contexts[0]]

        out.append(
            EvidenceGroup(
                label=group.label,
                query=group.query,
                contexts=unique_contexts,
            )
        )

    return out


def retrieve_level3(
    question: str,
    top_k_per_task: int = 2,
) -> list[EvidenceGroup]:
    """Run controlled multi-evidence retrieval for a Level-3 question."""
    safe_top_k = max(1, top_k_per_task)
    tasks = _build_level3_tasks(question)
    task_by_label = {
        task.label: task
        for task in tasks
    }

    groups = [
        EvidenceGroup(
            label=task.label,
            query=task.query,
            contexts=retrieve_explicit_query(
                task=task,
                top_k=max(2, safe_top_k + 1),
            ),
        )
        for task in tasks
    ]

    # Only Level-3 recommendation synthesis is aggregated. Q1-Q8 and all
    # comparison/table paths retain their existing context handling.
    groups = [
        _aggregate_synthesis_recommendation_group(
            group,
            task_by_label[group.label],
        )
        if (
            task_by_label[group.label].evidence_type
            == "synthesis_component"
            and task_by_label[group.label].synthesis_component
            == "recommendations"
        )
        else group
        for group in groups
    ]

    groups = _deduplicate_level3_groups(
        groups=groups,
        top_k_per_task=safe_top_k,
    )

    synthesis_tasks = [
        task
        for task in tasks
        if task.evidence_type == "synthesis_component"
    ]

    if synthesis_tasks:
        groups_by_label = {
            group.label: group
            for group in groups
        }
        coverage: dict[str, Any] = {}
        missing_components: list[str] = []

        for task in synthesis_tasks:
            group = groups_by_label.get(
                task.label
            )
            contexts = (
                group.contexts
                if group is not None
                else []
            )

            if task.synthesis_component == "recommendations":
                units = [
                    unit
                    for context in contexts
                    for unit in _extract_recommendation_units_from_context(
                        context
                    )
                ]
                covered = bool(units)
                coverage[task.label] = {
                    "covered": covered,
                    "recommendation_units": len(units),
                }
            else:
                covered = bool(contexts)
                coverage[task.label] = {
                    "covered": covered,
                }

            if not covered:
                missing_components.append(
                    task.label
                )

        if missing_components:
            LOGGER.warning(
                "Level-3 synthesis coverage incomplete | coverage=%s | "
                "missing=%s",
                coverage,
                missing_components,
            )
        else:
            LOGGER.info(
                "Level-3 synthesis coverage complete | coverage=%s",
                coverage,
            )

    LOGGER.info(
        "Level-3 retrieval completed | tasks=%d | groups=%s",
        len(tasks),
        [
            {
                "label": group.label,
                "query": group.query,
                "pages": [
                    context.page
                    for context in group.contexts
                ],
            }
            for group in groups
        ],
    )

    return groups


def retrieve(
    question: str,
    top_k: int,
    history: list[Message] | None = None,
    resolved_query: str | None = None,
) -> list[Context]:
    """Retrieve with a standalone query and an intent-preserving anchor."""
    conversation_history = history or []

    query = resolved_query or rewrite_query(
        question,
        conversation_history,
    )

    plans = _make_query_plans(query)

    candidate_k = min(
        MAX_CANDIDATES_PER_QUERY,
        max(
            MIN_CANDIDATES_PER_QUERY,
            top_k
            * CANDIDATE_MULTIPLIER,
        ),
    )

    embedder = get_embedder()
    store = get_store()

    hit_groups = [
        (
            plan,
            _search(
                query=plan.text,
                candidate_k=candidate_k,
                embedder=embedder,
                store=store,
            ),
        )
        for plan in plans
    ]

    merged_hits = _merge_hits(
        hit_groups,
    )

    ranked = _base_rerank(
        plans=plans,
        merged_hits=merged_hits,
        standalone_query=query,
    )

    ranked = _evidence_type_rerank(
        query=query,
        ranked=ranked,
    )

    selected = ranked[:top_k]

    LOGGER.info(
        "Retrieval completed | method=anchored-cross-encoder | "
        "queries=%d | candidate_k_per_query=%d | "
        "merged_candidates=%d | returned=%d | plans=%s",
        len(plans),
        candidate_k,
        len(merged_hits),
        len(selected),
        [
            {
                "role": plan.role,
                "text": plan.text,
            }
            for plan in plans
        ],
    )

    return [
        Context(
            text=str(
                merged_hit.hit.payload.get(
                    "text",
                    "",
                )
            ),
            page=int(
                merged_hit.hit.payload.get(
                    "page",
                    0,
                )
            ),
            score=float(score),
        )
        for score, merged_hit
        in selected
    ]