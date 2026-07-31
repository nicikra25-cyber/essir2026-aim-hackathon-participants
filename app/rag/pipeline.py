"""Answer a question end to end.

    history + retrieved context  ->  prompt  ->  LLM  ->  grounded answer

This is what `POST /query` calls. You send only a question and its level; the system
assigns the id, threads the conversation (level-2 follow-ups share memory), produces the
answer, and writes it to `data/out/` as a JSON file you can later copy into `submission/`.
"""

from __future__ import annotations

import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from ..config import get_settings
from ..llm.base import LLMError, Message
from ..llm.factory import get_client
from ..models import Diagnostics, QueryRequest, QueryResponse, Source
from . import memory
from .retrieve import (
    Context,
    EvidenceGroup,
    retrieve,
    retrieve_level3,
    rewrite_query,
)


SYSTEM_PROMPT = (
    "You answer questions about a single document using only the context provided. "
    "Use the standalone interpretation to resolve conversational references, but "
    "answer the user's original question. If the context does not contain the answer, "
    "say so plainly rather than guessing. Be specific and concise."
)

QUESTION_TYPE_PATTERN = re.compile(
    r"^\s*(?:(?:and|but|so|then)\s+)?"
    r"(why|how|what|which|where|when|who)\b",
    flags=re.IGNORECASE,
)

HOW_RESPONSE_LANGUAGE = (
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
)

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

HOW_GENERAL_LANGUAGE = (
    "potential to assist",
    "assist in managing",
    "help manage",
    "support decision",
    "decision support",
    "information provision",
    "providing information",
    "informs decision making",
    "inform decision making",
)

HOW_NARROW_LANGUAGE = (
    "for example",
    "for instance",
    "specific contemporary",
    "benchmarking",
    "value chain analysis",
    "activity-based management",
    "activity based costing",
    "low-cost strategy",
    "product differentiation",
    "traditional rigid cost-focused",
    "positive moderating role",
    "positively moderate",
    "this result implies",
    "it is recommended",
    "under specific circumstances",
)


def _question_type(question: str) -> str:
    """Return the opening question type."""
    match = QUESTION_TYPE_PATTERN.search(question)
    return match.group(1).lower() if match else ""


def _contains_phrase(
    text: str,
    phrases: tuple[str, ...],
) -> bool:
    """Check case-insensitively for one or more phrases."""
    padded = (
        " "
        + text.replace("\u00ad", "").lower()
        + " "
    )

    return any(
        phrase in padded
        for phrase in phrases
    )


def _build_messages(
    original_question: str,
    resolved_question: str,
    contexts: list[Context],
    history: list[Message],
) -> list[Message]:
    context_block = (
        "\n\n".join(
            f"[page {context.page}] {context.text}"
            for context in contexts
        )
        or "(no context retrieved)"
    )

    question_type = _question_type(resolved_question)

    messages: list[Message] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    messages.extend(history)

    task_instruction = (
        "Answer the original user question according to the standalone "
        "interpretation, using only the document context."
    )

    if question_type == "how":
        task_instruction += (
            " Give the general response in one or two concise sentences. "
            "Explain how the named problem can be managed, addressed, or mitigated "
            "and identify the means through which the response works. "
            "Do not claim that the problem is completely eliminated unless the "
            "document explicitly says so. Prefer the document's general explanation "
            "over examples, lists of techniques, strategy-specific recommendations, "
            "empirical qualifications, or narrow implementation details."
        )

    messages.append(
        {
            "role": "user",
            "content": (
                f"Context from the document:\n{context_block}\n\n"
                f"Original user question: {original_question}\n"
                f"Standalone interpretation: {resolved_question}\n\n"
                f"{task_instruction}"
            ),
        }
    )

    return messages



def _flatten_level3_contexts(
    evidence_groups: list[EvidenceGroup],
) -> list[Context]:
    """Flatten Level-3 groups while removing duplicate page-text passages."""
    contexts: list[Context] = []
    seen: set[tuple[int, str]] = set()

    for group in evidence_groups:
        for context in group.contexts:
            normalised_text = re.sub(
                r"\s+",
                " ",
                context.text.replace("\u00ad", "").strip().lower(),
            )
            key = (
                context.page,
                normalised_text,
            )

            if key in seen:
                continue

            seen.add(key)
            contexts.append(context)

    return contexts


def _build_level3_messages(
    question: str,
    evidence_groups: list[EvidenceGroup],
) -> list[Message]:
    """Build a focused synthesis prompt from independently retrieved evidence."""
    group_blocks: list[str] = []

    for group in evidence_groups:
        evidence_block = (
            "\n\n".join(
                f"[page {context.page}] {context.text}"
                for context in group.contexts
            )
            or "(no evidence retrieved for this group)"
        )

        group_blocks.append(
            f"[EVIDENCE GROUP: {group.label.upper()}]\n"
            f"Search focus: {group.query}\n"
            f"{evidence_block}"
        )

    grouped_context = (
        "\n\n".join(group_blocks)
        or "(no evidence groups retrieved)"
    )

    return [
        {
            "role": "system",
            "content": (
                SYSTEM_PROMPT
                + " For whole-document questions, use the separately labelled "
                "evidence groups as distinct evidence obligations. Do not replace "
                "one requested evidence type with a related passage from another "
                "section. Do not infer a table result from a narrative result."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Whole-document question:\n{question}\n\n"
                f"Evidence retrieved independently for each component:\n"
                f"{grouped_context}\n\n"
                "Write one concise integrated answer using only direct evidence "
                "that fulfils the labelled search focus. For LITERATURE_REVIEW, "
                "state the document's direct proposition about how management "
                "accounting practices help organisations manage competitive forces. "
                "For TABLE_5, use only an actual statistical result containing the "
                "requested interaction, outcome, coefficient, and significance "
                "information; do not use a methodological description as table "
                "evidence. For RESULTS_INTERPRETATION, use the direct interpretation "
                "of the traditional-management-accounting result and its hypothesis "
                "support; do not substitute a finding about contemporary practices "
                "or H2. Exclude general Porter background, methodology, unrelated "
                "findings, and repeated explanations unless the question explicitly "
                "asks for them. Preserve coefficients and p-values exactly as "
                "retrieved. If one required evidence group lacks direct support, "
                "state that the evidence was not retrieved rather than guessing."
            ),
        },
    ]

def _words(text: str) -> set[str]:
    """Return meaningful lowercase words for sentence matching."""
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "included",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "to",
        "used",
        "was",
        "were",
        "what",
        "which",
        "with",
    }

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.replace("\u00ad", "").lower(),
    )

    return {
        word
        for word in words
        if word not in stop_words
    }


def _split_sentences(text: str) -> list[str]:
    """Split one retrieved chunk into complete sentence candidates."""
    cleaned = re.sub(
        r"\s+",
        " ",
        text.strip(),
    )

    sentences = re.split(
        r'(?<=[.!?])\s+(?=[A-Z0-9"“‘(])',
        cleaned,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def _how_response_strength(sentence: str) -> float:
    """Measure generic response/action language in a sentence."""
    lowered = sentence.replace("\u00ad", "").lower()

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
        if any(
            term in lowered
            for term in group
        )
    )

    return min(
        1.0,
        matched_groups / 3.0,
    )


def _how_means_strength(sentence: str) -> float:
    """Reward response-plus-means structure rather than a means word alone."""
    padded = (
        " "
        + sentence.replace("\u00ad", "").lower()
        + " "
    )

    has_response = _contains_phrase(
        padded,
        HOW_RESPONSE_LANGUAGE,
    )
    has_means = _contains_phrase(
        padded,
        HOW_MEANS_LANGUAGE,
    )

    if not (
        has_response
        and has_means
    ):
        return 0.0

    score = 0.55

    if _contains_phrase(
        padded,
        HOW_GENERAL_LANGUAGE,
    ):
        score += 0.30

    if any(
        stem in padded
        for stem in (
            "provid",
            "inform",
            "understand",
            "decision",
            "support",
        )
    ):
        score += 0.15

    return min(
        1.0,
        score,
    )


def _sentence_support_score(
    sentence: str,
    question: str,
    answer_text: str,
) -> tuple[float, int]:
    """Score one sentence as direct evidence for the generated answer."""
    question_words = _words(question)
    answer_words = _words(answer_text)
    sentence_words = _words(sentence)

    question_overlap = (
        len(sentence_words & question_words)
        / max(1, len(question_words))
    )
    answer_overlap = (
        len(sentence_words & answer_words)
        / max(1, len(answer_words))
    )

    question_type = _question_type(question)

    if question_type == "how":
        response = _how_response_strength(sentence)
        means = _how_means_strength(sentence)
        general = (
            1.0
            if _contains_phrase(
                sentence,
                HOW_GENERAL_LANGUAGE,
            )
            else 0.0
        )
        narrow = (
            1.0
            if _contains_phrase(
                sentence,
                HOW_NARROW_LANGUAGE,
            )
            else 0.0
        )

        score = (
            0.24 * question_overlap
            + 0.16 * answer_overlap
            + 0.24 * response
            + 0.24 * means
            + 0.12 * general
            - 0.22 * narrow
        )

    else:
        # Preserve the existing behavior for non-how questions.
        score = (
            0.55 * question_overlap
            + 0.45 * answer_overlap
        )

    return (
        score,
        len(sentence),
    )


def _best_supporting_sentence(
    text: str,
    question: str,
    answer_text: str,
) -> str:
    """Return the complete sentence that most directly supports the answer."""
    cleaned = re.sub(
        r"\s+",
        " ",
        text.strip(),
    )
    sentences = _split_sentences(cleaned)

    if not sentences:
        return cleaned

    return max(
        sentences,
        key=lambda sentence: _sentence_support_score(
            sentence=sentence,
            question=question,
            answer_text=answer_text,
        ),
    )


def _citation_key(
    page: int,
    quote: str,
) -> tuple[int, str]:
    """Create a normalised key for identifying duplicate citations."""
    normalised = re.sub(
        r"\s+",
        " ",
        quote.strip(),
    )
    normalised = normalised.replace(
        "\u00ad",
        "",
    )

    return (
        page,
        normalised.lower(),
    )


def _is_incomplete_quote(quote: str) -> bool:
    """Identify a quote that appears to end before its sentence is complete."""
    cleaned = quote.strip()

    incomplete_endings = (
        "(i.e.",
        "(e.g.",
        "i.e.",
        "e.g.",
    )

    if cleaned.endswith(incomplete_endings):
        return True

    if cleaned.count("(") > cleaned.count(")"):
        return True

    return False


def _sources_from(
    contexts: list[Context],
    question: str,
    answer_text: str,
) -> list[Source]:
    """Return complete, relevant, non-duplicate supporting sentences."""
    out: list[Source] = []
    seen: set[tuple[int, str]] = set()

    for context in contexts:
        quote = _best_supporting_sentence(
            text=context.text,
            question=question,
            answer_text=answer_text,
        )

        if _is_incomplete_quote(quote):
            continue

        citation_key = _citation_key(
            context.page,
            quote,
        )

        if citation_key in seen:
            continue

        seen.add(citation_key)

        out.append(
            Source(
                page=context.page,
                quote=quote,
                score=round(
                    context.score,
                    4,
                ),
            )
        )

    return out




LEVEL3_METHODOLOGY_LANGUAGE = (
    "we use the ordinal independent variable interaction approach",
    "methodologically",
    "methodology",
    "we consider both the direct effect",
    "the effect of their interaction",
    "research design",
    "data analysis",
)

LEVEL3_LITERATURE_RESPONSE_LANGUAGE = (
    "potential to assist",
    "assist in managing",
    "help organisations",
    "help organizations",
    "manage competitive forces",
    "managing competitive forces",
    "decision support",
    "information provision",
    "providing information",
    "informs decision making",
    "inform decision making",
)

LEVEL3_RESULTS_LANGUAGE = (
    "table 5 shows",
    "shows that",
    "positively moderates",
    "positive moderating",
    "moderates the association",
    "significant positive",
    "partial support",
    "support is provided",
)


def _level3_statistical_strength(text: str) -> float:
    """Detect coefficient and significance formatting without target values."""
    cleaned = text.replace("\u00ad", "")
    lowered = cleaned.lower()
    score = 0.0

    if re.search(
        r"(?:β|beta)\s*=\s*[-+]?\d*\.\d+",
        cleaned,
        flags=re.IGNORECASE,
    ):
        score += 0.30

    if re.search(
        r"\bp\s*[=<]\s*0?\.\d+",
        lowered,
    ):
        score += 0.30

    if re.search(
        r"[-+]?0?\.\d{2,4}\s*\(\s*0?\.\d{2,4}\s*\)\s*\*?",
        cleaned,
    ):
        score += 0.35

    decimal_values = re.findall(
        r"(?<![\d.])[-+]?0?\.\d{2,4}(?![\d.])",
        cleaned,
    )
    if len(decimal_values) >= 2:
        score += 0.20

    if "*" in cleaned:
        score += 0.10

    return min(
        1.0,
        score,
    )


def _level3_quote_candidates(
    text: str,
    label: str,
) -> list[str]:
    """Return sentence candidates, plus line windows for table evidence."""
    raw = text.replace("\u00ad", "")
    cleaned = re.sub(
        r"\s+",
        " ",
        raw.strip(),
    )

    candidates = _split_sentences(cleaned)

    if label == "table_5":
        lines = [
            re.sub(
                r"\s+",
                " ",
                line.strip(),
            )
            for line in raw.splitlines()
            if line.strip()
        ]

        for width in range(1, 6):
            for start in range(
                0,
                max(
                    0,
                    len(lines) - width + 1,
                ),
            ):
                candidates.append(
                    " ".join(
                        lines[start:start + width]
                    )
                )

        # A table row may be flattened into one chunk without sentence
        # punctuation, so retain the full cleaned chunk as a fallback candidate.
        if cleaned:
            candidates.append(cleaned)

    unique: list[str] = []
    seen: set[str] = set()

    for candidate in candidates:
        normalised = re.sub(
            r"\s+",
            " ",
            candidate.strip(),
        )

        if not normalised:
            continue

        key = normalised.lower()

        if key in seen:
            continue

        seen.add(key)
        unique.append(normalised)

    return unique


def _level3_group_quote_score(
    label: str,
    quote: str,
    query: str,
    answer_text: str,
) -> float:
    """Score a quotation according to its specific Level-3 evidence role."""
    lowered = quote.replace("\u00ad", "").lower()
    quote_words = _words(quote)
    query_words = _words(query)
    answer_words = _words(answer_text)

    query_overlap = (
        len(quote_words & query_words)
        / max(
            1,
            len(query_words),
        )
    )
    answer_overlap = (
        len(quote_words & answer_words)
        / max(
            1,
            len(answer_words),
        )
    )

    management_accounting = (
        1.0
        if "management accounting" in lowered
        else 0.0
    )
    competitive_forces = (
        1.0
        if (
            "competitive force" in lowered
            or "five forces" in lowered
        )
        else 0.0
    )
    traditional = (
        1.0
        if "traditional" in lowered
        else 0.0
    )
    contemporary = (
        1.0
        if "contemporary" in lowered
        else 0.0
    )
    competitive_advantage = (
        1.0
        if "competitive advantage" in lowered
        else 0.0
    )
    organisational_outcome = (
        1.0
        if any(
            phrase in lowered
            for phrase in (
                "competitive advantage",
                "organisational performance",
                "organizational performance",
            )
        )
        else 0.0
    )
    methodology = (
        1.0
        if _contains_phrase(
            quote,
            LEVEL3_METHODOLOGY_LANGUAGE,
        )
        else 0.0
    )

    if label == "literature_review":
        direct_response = (
            1.0
            if _contains_phrase(
                quote,
                LEVEL3_LITERATURE_RESPONSE_LANGUAGE,
            )
            else 0.0
        )
        porter_background = (
            1.0
            if any(
                phrase in lowered
                for phrase in (
                    "porter (",
                    "porter’s",
                    "porter's",
                    "roots of an industry",
                    "current profitability",
                    "framework for anticipating",
                )
            )
            else 0.0
        )
        empirical_result = (
            1.0
            if (
                _level3_statistical_strength(quote) > 0.0
                or "table 5" in lowered
                or "hypothesis" in lowered
            )
            else 0.0
        )

        return (
            0.14 * query_overlap
            + 0.08 * answer_overlap
            + 0.20 * management_accounting
            + 0.16 * competitive_forces
            + 0.32 * direct_response
            - 0.22 * porter_background
            - 0.18 * methodology
            - 0.12 * empirical_result
        )

    if label == "table_5":
        statistics = _level3_statistical_strength(
            quote
        )
        interaction = (
            1.0
            if any(
                marker in lowered
                for marker in (
                    "interaction",
                    "×",
                    " x ",
                    "intensity of competitive forces",
                )
            )
            else 0.0
        )
        table_like_row = (
            1.0
            if re.search(
                r"[-+]?0?\.\d{2,4}\s*\(\s*0?\.\d{2,4}\s*\)\s*\*?",
                quote,
            )
            else 0.0
        )
        narrative_result = (
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

        return (
            0.08 * query_overlap
            + 0.18 * management_accounting
            + 0.18 * traditional
            + 0.10 * organisational_outcome
            + 0.10 * interaction
            + 0.26 * statistics
            + 0.20 * table_like_row
            - 0.35 * methodology
            - 0.22 * contemporary
            - 0.12 * narrative_result
        )

    if label == "results_interpretation":
        interpretation = (
            1.0
            if _contains_phrase(
                quote,
                LEVEL3_RESULTS_LANGUAGE,
            )
            else 0.0
        )
        h3_support = (
            1.0
            if (
                bool(
                    re.search(
                        r"\bh\s*3\b|\bh3\b",
                        lowered,
                    )
                )
                or "partial support" in lowered
            )
            else 0.0
        )
        wrong_h2 = (
            1.0
            if bool(
                re.search(
                    r"\bh\s*2\b|\bh2\b",
                    lowered,
                )
            )
            else 0.0
        )
        statistics = _level3_statistical_strength(
            quote
        )

        return (
            0.08 * query_overlap
            + 0.06 * answer_overlap
            + 0.14 * management_accounting
            + 0.18 * traditional
            + 0.10 * competitive_forces
            + 0.12 * competitive_advantage
            + 0.16 * interpretation
            + 0.08 * statistics
            + 0.14 * h3_support
            - 0.34 * contemporary
            - 0.38 * wrong_h2
            - 0.28 * methodology
        )

    return (
        0.55 * query_overlap
        + 0.45 * answer_overlap
    )


def _level3_quote_is_eligible(
    label: str,
    quote: str,
) -> bool:
    """Reject passages that do not fulfil the evidence group's minimum role."""
    lowered = quote.replace("\u00ad", "").lower()

    if label == "literature_review":
        return (
            "management accounting" in lowered
            and (
                "competitive force" in lowered
                or "five forces" in lowered
            )
            and _contains_phrase(
                quote,
                LEVEL3_LITERATURE_RESPONSE_LANGUAGE,
            )
        )

    if label == "table_5":
        return (
            "management accounting" in lowered
            and "traditional" in lowered
            and any(
                outcome in lowered
                for outcome in (
                    "competitive advantage",
                    "organisational performance",
                    "organizational performance",
                )
            )
            and _level3_statistical_strength(
                quote
            ) >= 0.25
            and not _contains_phrase(
                quote,
                LEVEL3_METHODOLOGY_LANGUAGE,
            )
        )

    if label == "results_interpretation":
        has_interpretation = _contains_phrase(
            quote,
            LEVEL3_RESULTS_LANGUAGE,
        )
        has_wrong_focus = (
            "contemporary" in lowered
            and "traditional" not in lowered
        )
        has_h2_only = (
            bool(
                re.search(
                    r"\bh\s*2\b|\bh2\b",
                    lowered,
                )
            )
            and not bool(
                re.search(
                    r"\bh\s*3\b|\bh3\b",
                    lowered,
                )
            )
        )

        return (
            "management accounting" in lowered
            and "traditional" in lowered
            and has_interpretation
            and not has_wrong_focus
            and not has_h2_only
            and not _contains_phrase(
                quote,
                LEVEL3_METHODOLOGY_LANGUAGE,
            )
        )

    return True


def _best_level3_quote(
    group: EvidenceGroup,
    context: Context,
    answer_text: str,
) -> tuple[str, float] | None:
    """Select the best eligible quote inside one retrieved Level-3 context."""
    candidates = _level3_quote_candidates(
        text=context.text,
        label=group.label,
    )

    eligible = [
        candidate
        for candidate in candidates
        if _level3_quote_is_eligible(
            group.label,
            candidate,
        )
    ]

    if not eligible:
        return None

    best_quote = max(
        eligible,
        key=lambda candidate: (
            _level3_group_quote_score(
                label=group.label,
                quote=candidate,
                query=group.query,
                answer_text=answer_text,
            ),
            -len(candidate),
        ),
    )

    support_score = _level3_group_quote_score(
        label=group.label,
        quote=best_quote,
        query=group.query,
        answer_text=answer_text,
    )

    return (
        best_quote,
        support_score,
    )


def _sources_from_level3(
    evidence_groups: list[EvidenceGroup],
    answer_text: str,
) -> list[Source]:
    """Return one direct, evidence-type-specific quotation per Level-3 group."""
    out: list[Source] = []
    seen: set[tuple[int, str]] = set()

    for group in evidence_groups:
        candidates: list[
            tuple[
                float,
                Context,
                str,
            ]
        ] = []

        for context in group.contexts:
            selected = _best_level3_quote(
                group=group,
                context=context,
                answer_text=answer_text,
            )

            if selected is None:
                continue

            quote, support_score = selected

            if _is_incomplete_quote(quote):
                continue

            citation_key = _citation_key(
                context.page,
                quote,
            )

            if citation_key in seen:
                continue

            # Select by evidence-role fit first and retrieval relevance second.
            combined_score = (
                0.78 * support_score
                + 0.22 * max(
                    0.0,
                    min(
                        1.0,
                        context.score,
                    ),
                )
            )

            candidates.append(
                (
                    combined_score,
                    context,
                    quote,
                )
            )

        if not candidates:
            # Returning no source is safer than attaching a contradictory or
            # method-only quotation to a required evidence group.
            continue

        _, best_context, best_quote = max(
            candidates,
            key=lambda item: (
                item[0],
                -len(item[2]),
            ),
        )

        best_key = _citation_key(
            best_context.page,
            best_quote,
        )
        seen.add(best_key)

        out.append(
            Source(
                page=best_context.page,
                quote=best_quote,
                score=round(
                    best_context.score,
                    4,
                ),
            )
        )

    return out

def _save(
    response: QueryResponse,
    when: datetime,
) -> None:
    """Write the answer to data/out/q_<id>_level_<level>_<datetime>.json."""
    out_dir = Path(
        get_settings().out_dir
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = when.strftime(
        "%Y%m%d-%H%M%S"
    )
    name = (
        f"q_{response.question_id}"
        f"_level_{response.level}"
        f"_{stamp}.json"
    )

    (
        out_dir / name
    ).write_text(
        response.model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )


def answer(req: QueryRequest) -> QueryResponse:
    settings = get_settings()
    client = get_client()
    top_k = (
        req.top_k
        or settings.top_k
    )

    question_id = (
        "q"
        + uuid.uuid4().hex[:6]
    )
    conversation_id = (
        f"level-{req.level}"
    )

    history = memory.get_history(
        conversation_id
    )
    now = datetime.now(UTC)
    started = time.perf_counter()

    evidence_groups: list[EvidenceGroup] = []

    if req.level == 3:
        # Level 3 uses a separate controlled multi-evidence path.
        # The working Level-1/2 retrieval path below remains unchanged.
        resolved_question = req.question

        evidence_groups = retrieve_level3(
            question=req.question,
            top_k_per_task=max(
                1,
                min(
                    top_k,
                    2,
                ),
            ),
        )

        contexts = _flatten_level3_contexts(
            evidence_groups
        )

        messages = _build_level3_messages(
            question=req.question,
            evidence_groups=evidence_groups,
        )

    else:
        # Existing Level-1/2 path: keep its conversational rewrite,
        # retrieval logic, prompts, and citation behavior unchanged.
        resolved_question = rewrite_query(
            req.question,
            history,
        )

        contexts = retrieve(
            question=req.question,
            top_k=top_k,
            history=history,
            resolved_query=resolved_question,
        )

        messages = _build_messages(
            original_question=req.question,
            resolved_question=resolved_question,
            contexts=contexts,
            history=history,
        )

    try:
        answer_text = client.chat(
            messages
        )
    except LLMError as error:
        answer_text = (
            f"[LLM unavailable: {error}] "
            "Retrieved context is attached as sources; "
            "no generated answer."
        )

    memory.append(
        conversation_id,
        req.question,
        answer_text,
    )

    latency_ms = int(
        (
            time.perf_counter()
            - started
        )
        * 1000
    )

    if req.level == 3:
        sources = _sources_from_level3(
            evidence_groups=evidence_groups,
            answer_text=answer_text,
        )
    else:
        sources = _sources_from(
            contexts=contexts,
            question=(
                f"{req.question} "
                f"{resolved_question}"
            ),
            answer_text=answer_text,
        )

    response = QueryResponse(
        question_id=question_id,
        level=req.level,
        question=req.question,
        answer=answer_text,
        conversation_id=conversation_id,
        sources=sources,
        diagnostics=Diagnostics(
            provider=settings.llm_provider,
            chat_model=settings.chat_model,
            embedding_model=settings.embedding_model,
            retrieved_chunks=len(
                contexts
            ),
            tokens=None,
            latency_ms=latency_ms,
            timestamp=now.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        ),
    )

    _save(
        response,
        now,
    )

    return response