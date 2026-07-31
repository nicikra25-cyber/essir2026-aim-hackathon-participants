"""Load a PDF into the vector store.

    parse PDF (data/in) -> pages -> [chunk] -> embeddings -> Qdrant

Normal pages are extracted with PyMuPDF, chunked in chunking.py, and embedded
using a local sentence-transformers model.

A separate Level-3-only helper can extract tables with pdfplumber and store them
in a physically separate Qdrant collection. Normal ingestion and Level-1/2
retrieval remain PyMuPDF-only.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pymupdf
from qdrant_client import models

from ..config import get_settings
from ..models import IngestResponse
from ..vectorstore.qdrant_store import get_store
from .chunking import chunk_pages
from .embeddings import get_embedder


# A fixed namespace so re-ingesting the same document overwrites its normal
# PyMuPDF points instead of duplicating them.
_NAMESPACE = uuid.UUID(
    "6f0d9b1e-3b7a-4c2e-9a1d-000000000000"
)

# A different namespace for Level-3 pdfplumber table points.
_TABLE_NAMESPACE = uuid.UUID(
    "6f0d9b1e-3b7a-4c2e-9a1d-111111111111"
)

# Used only by the Level-3 table helper when an unruled table must be inferred
# from word positions rather than visible cell borders.
_TEXT_TABLE_SETTINGS = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "min_words_vertical": 2,
    "min_words_horizontal": 1,
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "intersection_tolerance": 5,
    "text_x_tolerance": 2,
    "text_y_tolerance": 2,
}


def _find_pdf(filename: str | None) -> Path:
    """Find the requested PDF, or the first PDF in the input directory."""
    in_dir = Path(get_settings().in_dir)

    if filename:
        path = in_dir / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"no such PDF: {path}"
            )

        return path

    pdfs = sorted(
        in_dir.glob("*.pdf")
    )

    if not pdfs:
        raise FileNotFoundError(
            f"no *.pdf found in {in_dir}/ — "
            "put your document there first"
        )

    return pdfs[0]


def extract_pages(path: Path) -> list[str]:
    """Extract one text string per PDF page using PyMuPDF.

    PyMuPDF was selected after comparison with pypdf and pdfplumber because it
    produced cleaner narrative text and fewer broken words for this document.

    Pages remain separate so every chunk retains an unambiguous PDF page number
    for citations. ``sort=True`` attempts to preserve a natural reading order.
    """
    with pymupdf.open(
        str(path)
    ) as document:
        return [
            page.get_text(
                "text",
                sort=True,
            )
            or ""
            for page in document
        ]


def ingest(
    filename: str | None = None,
    reset: bool = False,
) -> IngestResponse:
    """Run the normal PyMuPDF ingestion used by Levels 1 and 2."""
    settings = get_settings()
    embedder = get_embedder()
    store = get_store()

    path = _find_pdf(filename)
    pages = extract_pages(path)
    chunks = chunk_pages(pages)

    if not chunks:
        raise ValueError(
            f"{path.name} produced no text — "
            "is it a scanned/image PDF?"
        )

    # Embed in batches. is_query=False marks these as documents
    # ("passage:" for the E5 embedding model).
    vectors: list[list[float]] = []
    batch_size = 32

    for start in range(
        0,
        len(chunks),
        batch_size,
    ):
        batch_chunks = chunks[
            start : start + batch_size
        ]
        texts = [
            chunk.text
            for chunk in batch_chunks
        ]

        vectors.extend(
            embedder.embed(
                texts,
                is_query=False,
            )
        )

    store.ensure_collection(
        dim=len(vectors[0]),
        reset=reset,
    )

    points = [
        models.PointStruct(
            id=str(
                uuid.uuid5(
                    _NAMESPACE,
                    f"{path.name}:{chunk.index}",
                )
            ),
            vector=vector,
            payload={
                "text": chunk.text,
                "page": chunk.page,
                "source": path.name,
                "content_type": "narrative",
                "extractor": "pymupdf",
            },
        )
        for chunk, vector in zip(
            chunks,
            vectors,
        )
    ]

    store.upsert(points)

    return IngestResponse(
        document=path.name,
        pages=len(pages),
        chunks=len(chunks),
        collection=settings.qdrant_collection,
    )


_TABLE_INDEX_VERSION = 3

# Used only after a page has been identified as containing rotated text.
_TEXT_TABLE_SETTINGS = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "min_words_vertical": 2,
    "min_words_horizontal": 1,
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "intersection_tolerance": 5,
    "text_x_tolerance": 2,
    "text_y_tolerance": 2,
}

_READABLE_TABLE_WORDS = {
    "accounting",
    "advantage",
    "coefficient",
    "competitive",
    "forces",
    "goodness",
    "management",
    "model",
    "organisational",
    "organizational",
    "overall",
    "path",
    "performance",
    "sample",
    "significance",
    "strategy",
    "table",
    "traditional",
}

_TABLE_STOP_WORDS = {
    "and",
    "for",
    "from",
    "into",
    "model",
    "overall",
    "page",
    "sample",
    "table",
    "that",
    "the",
    "this",
    "with",
}


def _page_contains_rotated_text(page: Any) -> bool:
    """Detect a page containing a substantial amount of sideways text."""
    chars = [
        char
        for char in getattr(page, "chars", [])
        if str(char.get("text", "")).strip()
    ]

    if not chars:
        return False

    rotated = sum(
        1
        for char in chars
        if not bool(char.get("upright", True))
    )

    return rotated / len(chars) >= 0.20


def _median(values: list[float]) -> float:
    """Return the median without importing another module."""
    if not values:
        return 0.0

    ordered = sorted(values)
    midpoint = len(ordered) // 2

    if len(ordered) % 2:
        return float(ordered[midpoint])

    return float(
        (
            ordered[midpoint - 1]
            + ordered[midpoint]
        )
        / 2.0
    )


def _numeric_count(text: str) -> int:
    """Count integer and decimal values in table text."""
    return len(
        re.findall(
            r"(?<![\w.])[-+]?(?:\d+\.\d+|\d+)(?![\w.])",
            text,
        )
    )


def _clean_table(
    table: list[list[Any]] | None,
) -> list[list[str]]:
    """Normalise one extracted table and remove empty rows."""
    rows: list[list[str]] = []

    for row in table or []:
        if not row:
            continue

        cleaned = [
            re.sub(
                r"\s+",
                " ",
                str(cell or "").strip(),
            )
            for cell in row
        ]

        if any(cleaned):
            rows.append(cleaned)

    return rows


def _is_probable_table(
    table: list[list[Any]] | None,
) -> bool:
    """Reject normal prose that pdfplumber has split into fake columns."""
    rows = _clean_table(table)

    if len(rows) < 2:
        return False

    width = max(
        len(row)
        for row in rows
    )

    if width < 2:
        return False

    non_empty = [
        cell
        for row in rows
        for cell in row
        if cell
    ]

    multi_cell_rows = sum(
        1
        for row in rows
        if sum(bool(cell) for cell in row) >= 2
    )

    if (
        len(non_empty) < 4
        or multi_cell_rows < 2
    ):
        return False

    combined = " ".join(non_empty)

    if _numeric_count(combined) < 2:
        return False

    average_words = sum(
        len(re.findall(r"\b\w+\b", cell))
        for cell in non_empty
    ) / len(non_empty)

    if (
        average_words > 14
        and width <= 5
    ):
        return False

    return True


def _extract_valid_tables(
    page: Any,
    *,
    allow_text_strategy: bool,
) -> list[list[list[Any]]]:
    """Extract tables without applying permissive text rules to prose pages."""
    attempts: list[list[list[list[Any]]]] = [
        page.extract_tables() or [],
    ]

    if allow_text_strategy:
        attempts.append(
            page.extract_tables(
                table_settings=_TEXT_TABLE_SETTINGS
            )
            or []
        )

    for candidates in attempts:
        valid = [
            table
            for table in candidates
            if _is_probable_table(table)
        ]

        if valid:
            return valid

    return []


def _normalise_word_text(
    text: str,
    reverse_token: bool,
) -> str:
    """Normalise one word and optionally reverse its character direction."""
    cleaned = re.sub(
        r"\s+",
        " ",
        str(text or "").strip(),
    )

    if reverse_token:
        cleaned = cleaned[::-1]

    return cleaned


def _extract_words(
    page: Any,
    *,
    reverse_token: bool,
) -> list[dict[str, Any]]:
    """Extract words together with their PDF coordinates."""
    raw_words = page.extract_words(
        x_tolerance=1,
        y_tolerance=2,
        keep_blank_chars=False,
        use_text_flow=False,
    ) or []

    words: list[dict[str, Any]] = []

    for raw in raw_words:
        text = _normalise_word_text(
            str(raw.get("text", "")),
            reverse_token=reverse_token,
        )

        if not text:
            continue

        words.append(
            {
                "text": text,
                "x0": float(raw.get("x0", 0.0)),
                "x1": float(raw.get("x1", 0.0)),
                "top": float(raw.get("top", 0.0)),
                "bottom": float(raw.get("bottom", 0.0)),
            }
        )

    return words


def _group_words_into_physical_rows(
    words: list[dict[str, Any]],
    *,
    descending_x: bool,
    reverse_row_order: bool,
) -> list[dict[str, Any]]:
    """Group words with similar y coordinates into physical rows."""
    if not words:
        return []

    heights = [
        max(
            1.0,
            float(word["bottom"])
            - float(word["top"]),
        )
        for word in words
    ]
    y_tolerance = max(
        2.0,
        0.60 * _median(heights),
    )

    ordered_words = sorted(
        words,
        key=lambda word: (
            (
                float(word["top"])
                + float(word["bottom"])
            )
            / 2.0,
            float(word["x0"]),
        ),
    )

    grouped: list[dict[str, Any]] = []

    for word in ordered_words:
        center_y = (
            float(word["top"])
            + float(word["bottom"])
        ) / 2.0

        if (
            not grouped
            or abs(
                center_y
                - float(grouped[-1]["center_y"])
            )
            > y_tolerance
        ):
            grouped.append(
                {
                    "center_y": center_y,
                    "words": [word],
                }
            )
        else:
            group = grouped[-1]
            group["words"].append(word)
            count = len(group["words"])
            group["center_y"] = (
                (
                    float(group["center_y"])
                    * (count - 1)
                )
                + center_y
            ) / count

    if reverse_row_order:
        grouped.reverse()

    for group in grouped:
        group["words"].sort(
            key=lambda word: float(word["x0"]),
            reverse=descending_x,
        )

    return grouped


def _words_to_cells(
    row_words: list[dict[str, Any]],
    *,
    descending_x: bool,
) -> list[dict[str, Any]]:
    """Join adjacent words into cells while preserving coordinates."""
    if not row_words:
        return []

    character_widths = [
        (
            max(
                1.0,
                float(word["x1"])
                - float(word["x0"]),
            )
            / max(
                1,
                len(str(word["text"])),
            )
        )
        for word in row_words
    ]
    gap_tolerance = max(
        3.0,
        2.2 * _median(character_widths),
    )

    cells: list[dict[str, Any]] = []

    for word in row_words:
        if not cells:
            cells.append(
                {
                    "text_parts": [str(word["text"])],
                    "x0": float(word["x0"]),
                    "x1": float(word["x1"]),
                    "top": float(word["top"]),
                    "bottom": float(word["bottom"]),
                }
            )
            continue

        previous = cells[-1]

        if descending_x:
            gap = (
                float(previous["x0"])
                - float(word["x1"])
            )
        else:
            gap = (
                float(word["x0"])
                - float(previous["x1"])
            )

        if gap <= gap_tolerance:
            previous["text_parts"].append(
                str(word["text"])
            )
            previous["x0"] = min(
                float(previous["x0"]),
                float(word["x0"]),
            )
            previous["x1"] = max(
                float(previous["x1"]),
                float(word["x1"]),
            )
            previous["top"] = min(
                float(previous["top"]),
                float(word["top"]),
            )
            previous["bottom"] = max(
                float(previous["bottom"]),
                float(word["bottom"]),
            )
        else:
            cells.append(
                {
                    "text_parts": [str(word["text"])],
                    "x0": float(word["x0"]),
                    "x1": float(word["x1"]),
                    "top": float(word["top"]),
                    "bottom": float(word["bottom"]),
                }
            )

    out: list[dict[str, Any]] = []

    for cell in cells:
        out.append(
            {
                "text": " ".join(
                    str(part)
                    for part in cell["text_parts"]
                ),
                "x0": round(float(cell["x0"]), 3),
                "x1": round(float(cell["x1"]), 3),
                "top": round(float(cell["top"]), 3),
                "bottom": round(float(cell["bottom"]), 3),
                "center_x": round(
                    (
                        float(cell["x0"])
                        + float(cell["x1"])
                    )
                    / 2.0,
                    3,
                ),
            }
        )

    return out


def _cluster_column_anchors(
    rows: list[dict[str, Any]],
) -> list[float]:
    """Infer stable column positions from cell x coordinates."""
    centers = sorted(
        float(cell["center_x"])
        for row in rows
        for cell in row["cells"]
    )

    if not centers:
        return []

    widths = [
        max(
            1.0,
            float(cell["x1"])
            - float(cell["x0"]),
        )
        for row in rows
        for cell in row["cells"]
    ]
    tolerance = max(
        10.0,
        0.45 * _median(widths),
    )

    clusters: list[list[float]] = []

    for center in centers:
        if (
            not clusters
            or abs(
                center
                - (
                    sum(clusters[-1])
                    / len(clusters[-1])
                )
            )
            > tolerance
        ):
            clusters.append([center])
        else:
            clusters[-1].append(center)

    return [
        round(
            sum(cluster) / len(cluster),
            3,
        )
        for cluster in clusters
    ]


def _assign_cell_ids(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign row, column, and cell identifiers to reconstructed cells."""
    column_anchors = _cluster_column_anchors(
        rows
    )

    for row_index, row in enumerate(
        rows,
        start=1,
    ):
        row_id = f"row_{row_index}"
        row["row_id"] = row_id

        used_column_ids: dict[str, int] = {}

        for cell in row["cells"]:
            if column_anchors:
                nearest_index = min(
                    range(len(column_anchors)),
                    key=lambda index: abs(
                        float(cell["center_x"])
                        - column_anchors[index]
                    ),
                )
            else:
                nearest_index = 0

            column_id = (
                f"column_{nearest_index + 1}"
            )
            used_column_ids[column_id] = (
                used_column_ids.get(
                    column_id,
                    0,
                )
                + 1
            )

            occurrence = used_column_ids[
                column_id
            ]

            cell["column_id"] = column_id
            cell["cell_id"] = (
                f"cell_{row_index}_"
                f"{nearest_index + 1}"
                + (
                    f"_{occurrence}"
                    if occurrence > 1
                    else ""
                )
            )

    return rows


def _extract_keywords(text: str) -> list[str]:
    """Create searchable row keywords from reconstructed table text."""
    tokens = re.findall(
        r"[A-Za-z][A-Za-z-]{2,}",
        text.lower(),
    )

    return sorted(
        {
            token
            for token in tokens
            if token not in _TABLE_STOP_WORDS
        }
    )


def _extract_statistical_pairs(
    text: str,
) -> list[dict[str, str]]:
    """Keep a coefficient and its parenthesised p-value in the same row."""
    compact = re.sub(
        r"\s*\|\s*",
        " ",
        text,
    )

    pattern = re.compile(
        r"(?P<coefficient>[-+]?(?:\d+\.\d+|\.\d+))"
        r"\s*"
        r"\(\s*"
        r"(?P<p_value>(?:\d+\.\d+|\.\d+))"
        r"\s*\)"
        r"(?P<stars>\*{1,3})?"
    )

    return [
        {
            "coefficient": match.group(
                "coefficient"
            ),
            "p_value": match.group(
                "p_value"
            ),
            "significance_marker": (
                match.group("stars")
                or ""
            ),
        }
        for match in pattern.finditer(compact)
    ]


def _row_label(text: str) -> str:
    """Return the non-numeric descriptive portion of a reconstructed row."""
    label = re.sub(
        r"[-+]?(?:\d+\.\d+|\.\d+)",
        " ",
        text,
    )
    label = re.sub(
        r"[\(\)\*\|]+",
        " ",
        label,
    )

    return re.sub(
        r"\s+",
        " ",
        label,
    ).strip()


def _build_logical_rows(
    physical_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge wrapped labels with the numeric cells that belong to that row."""
    prepared_rows: list[dict[str, Any]] = []

    for physical_row in physical_rows:
        cells = physical_row["cells"]
        text = " | ".join(
            str(cell["text"])
            for cell in cells
            if str(cell["text"]).strip()
        )

        if not text:
            continue

        prepared_rows.append(
            {
                "text": text,
                "cells": cells,
                "numeric_count": _numeric_count(
                    text
                ),
                "center_y": physical_row[
                    "center_y"
                ],
            }
        )

    logical_rows: list[dict[str, Any]] = []
    pending_labels: list[dict[str, Any]] = []

    for row in prepared_rows:
        if int(row["numeric_count"]) == 0:
            pending_labels.append(row)
            pending_labels = pending_labels[
                -8:
            ]
            continue

        related_rows = (
            pending_labels
            + [row]
        )
        pending_labels = []

        combined_cells = [
            cell
            for related in related_rows
            for cell in related["cells"]
        ]
        combined_text = "\n".join(
            str(related["text"])
            for related in related_rows
        )

        if (
            not re.search(
                r"[A-Za-z]",
                combined_text,
            )
            or _numeric_count(
                combined_text
            )
            < 2
        ):
            continue

        logical_rows.append(
            {
                "text": combined_text,
                "cells": combined_cells,
                "numeric_count": (
                    _numeric_count(
                        combined_text
                    )
                ),
                "center_y": row["center_y"],
            }
        )

    return _assign_cell_ids(
        logical_rows
    )


def _coordinate_rows_from_page(
    page: Any,
) -> list[dict[str, Any]]:
    """Reconstruct row/column structure from word coordinates.

    Multiple reading-direction hypotheses are evaluated because rotated PDF
    text can be stored with reversed character or row order.
    """
    candidates: list[
        tuple[float, list[dict[str, Any]]]
    ] = []

    for reverse_token in (
        False,
        True,
    ):
        words = _extract_words(
            page,
            reverse_token=reverse_token,
        )

        for descending_x in (
            False,
            True,
        ):
            for reverse_row_order in (
                False,
                True,
            ):
                grouped = (
                    _group_words_into_physical_rows(
                        words,
                        descending_x=descending_x,
                        reverse_row_order=(
                            reverse_row_order
                        ),
                    )
                )

                physical_rows = [
                    {
                        "center_y": group[
                            "center_y"
                        ],
                        "cells": _words_to_cells(
                            group["words"],
                            descending_x=(
                                descending_x
                            ),
                        ),
                    }
                    for group in grouped
                ]

                logical_rows = (
                    _build_logical_rows(
                        physical_rows
                    )
                )

                if not logical_rows:
                    continue

                combined = "\n".join(
                    str(row["text"])
                    for row in logical_rows
                )

                readable = _readability_score(
                    combined
                )
                keyword_bonus = sum(
                    1
                    for phrase in (
                        "traditional management accounting",
                        "competitive advantage",
                        "organisational performance",
                        "organizational performance",
                        "intensity of competitive forces",
                    )
                    if phrase in combined.lower()
                )

                pair_count = sum(
                    len(
                        _extract_statistical_pairs(
                            str(row["text"])
                        )
                    )
                    for row in logical_rows
                )

                score = (
                    6.0 * readable
                    + 2.0 * keyword_bonus
                    + min(
                        8.0,
                        len(logical_rows)
                        / 2.0,
                    )
                    + min(
                        8.0,
                        pair_count,
                    )
                )

                candidates.append(
                    (
                        score,
                        logical_rows,
                    )
                )

    if not candidates:
        return []

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return candidates[0][1]


def _readability_score(text: str) -> float:
    """Prefer the rotation and direction that restore readable table text."""
    words = set(
        re.findall(
            r"[A-Za-z]{3,}",
            text.lower(),
        )
    )

    if not words:
        return 0.0

    matches = len(
        words & _READABLE_TABLE_WORDS
    )

    return min(
        1.0,
        matches / 6.0,
    )


def _upright_ratio(page: Any) -> float:
    """Measure how many visible characters are upright after rotation."""
    chars = [
        char
        for char in getattr(page, "chars", [])
        if str(char.get("text", "")).strip()
    ]

    if not chars:
        return 0.0

    return sum(
        1
        for char in chars
        if bool(char.get("upright", True))
    ) / len(chars)


def _extract_rotated_coordinate_rows(
    path: Path,
    page_number: int,
) -> tuple[list[dict[str, Any]], int | None]:
    """Rotate one page internally and reconstruct rows from word coordinates.

    The source PDF is never modified. Temporary one-page PDFs are deleted
    automatically after each attempted rotation.
    """
    # Lazy imports preserve the Level-3-only boundary.
    import pdfplumber
    from pypdf import PdfReader, PdfWriter

    candidates: list[
        tuple[
            float,
            list[dict[str, Any]],
            int,
        ]
    ] = []

    for angle in (
        90,
        270,
        180,
    ):
        reader = PdfReader(str(path))
        source_page = reader.pages[
            page_number - 1
        ]
        source_page.rotate(angle)

        writer = PdfWriter()
        writer.add_page(source_page)

        with TemporaryDirectory(
            prefix="level3_coordinate_table_"
        ) as temporary_directory:
            temporary_path = (
                Path(temporary_directory)
                / f"page_{page_number}_{angle}.pdf"
            )

            with temporary_path.open(
                "wb"
            ) as temporary_file:
                writer.write(temporary_file)

            with pdfplumber.open(
                str(temporary_path)
            ) as rotated_document:
                rotated_page = (
                    rotated_document.pages[0]
                    .dedupe_chars()
                )

                rows = (
                    _coordinate_rows_from_page(
                        rotated_page
                    )
                )

                if not rows:
                    continue

                combined = "\n".join(
                    str(row["text"])
                    for row in rows
                )

                pair_count = sum(
                    len(
                        _extract_statistical_pairs(
                            str(row["text"])
                        )
                    )
                    for row in rows
                )

                score = (
                    8.0 * _readability_score(
                        combined
                    )
                    + 2.0 * _upright_ratio(
                        rotated_page
                    )
                    + min(
                        8.0,
                        len(rows) / 2.0,
                    )
                    + min(
                        8.0,
                        pair_count,
                    )
                )

                candidates.append(
                    (
                        score,
                        rows,
                        angle,
                    )
                )

    if not candidates:
        return [], None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    _, rows, angle = candidates[0]

    return rows, angle


def _normal_table_rows(
    table: list[list[Any]] | None,
) -> list[dict[str, Any]]:
    """Convert a normally extracted table into the same row-record format."""
    cleaned = _clean_table(table)
    rows: list[dict[str, Any]] = []

    for row_index, row in enumerate(
        cleaned,
        start=1,
    ):
        text = " | ".join(row)

        if (
            not re.search(r"[A-Za-z]", text)
            or _numeric_count(text) < 2
        ):
            continue

        cells = [
            {
                "row_id": f"row_{row_index}",
                "column_id": (
                    f"column_{column_index}"
                ),
                "cell_id": (
                    f"cell_{row_index}_"
                    f"{column_index}"
                ),
                "text": cell,
                "x0": None,
                "x1": None,
                "top": None,
                "bottom": None,
            }
            for column_index, cell in enumerate(
                row,
                start=1,
            )
        ]

        rows.append(
            {
                "row_id": f"row_{row_index}",
                "text": text,
                "cells": cells,
                "numeric_count": _numeric_count(
                    text
                ),
            }
        )

    return rows


def _row_record_to_chunk(
    row: dict[str, Any],
    *,
    page_number: int,
    table_index: int,
    rotation_applied: int,
    extraction_mode: str,
) -> dict[str, object]:
    """Create one independently searchable table-row payload."""
    row_text = str(row["text"])
    row_id = str(row["row_id"])
    cells = list(row["cells"])
    statistics = _extract_statistical_pairs(
        row_text
    )
    keywords = _extract_keywords(
        row_text
    )
    label = _row_label(
        row_text
    )

    cell_summary = " | ".join(
        (
            f"{cell['cell_id']}="
            f"{cell['text']}"
        )
        for cell in cells
    )

    statistics_summary = "; ".join(
        (
            f"coefficient="
            f"{pair['coefficient']}, "
            f"p_value={pair['p_value']}"
            + (
                f", significance="
                f"{pair['significance_marker']}"
                if pair[
                    "significance_marker"
                ]
                else ""
            )
        )
        for pair in statistics
    )

    search_text_parts = [
        f"Table row {row_id}.",
        f"Row label: {label}.",
        f"Cells: {cell_summary}.",
    ]

    if statistics_summary:
        search_text_parts.append(
            f"Statistics from this same row: "
            f"{statistics_summary}."
        )

    return {
        "text": " ".join(
            search_text_parts
        ),
        "raw_row_text": row_text,
        "row_label": label,
        "row_id": row_id,
        "cells": cells,
        "keywords": keywords,
        "statistics": statistics,
        "page": page_number,
        "table_index": table_index,
        "rotation_applied": rotation_applied,
        "extraction_mode": extraction_mode,
    }


def _table_index_is_current(
    table_store: Any,
    source_name: str,
) -> bool:
    """Check whether the existing collection uses the coordinate extractor."""
    if (
        not table_store.exists()
        or table_store.count() == 0
    ):
        return False

    try:
        records, _ = table_store.client.scroll(
            collection_name=table_store.collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(
                            value=source_name
                        ),
                    ),
                    models.FieldCondition(
                        key="index_version",
                        match=models.MatchValue(
                            value=_TABLE_INDEX_VERSION
                        ),
                    ),
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )

        return bool(records)

    except Exception:
        return False


def ensure_level3_table_index(
    filename: str | None = None,
    reset: bool = False,
) -> None:
    """Create the separate coordinate-based Level-3 table index.

    Levels 1 and 2 remain PyMuPDF-only. Rotated pages are handled with temporary
    internal copies. Each reconstructed row is stored separately with stable
    row, column, and cell identifiers so a retrieval query can match the row
    label and keep its coefficient and p-value attached to that same row.
    """
    # Lazy imports preserve the Level-3-only boundary.
    import pdfplumber

    from ..vectorstore.qdrant_store import get_table_store

    path = _find_pdf(filename)
    table_store = get_table_store()

    if (
        not reset
        and _table_index_is_current(
            table_store,
            path.name,
        )
    ):
        return

    embedder = get_embedder()
    table_chunks: list[
        dict[str, object]
    ] = []

    with pdfplumber.open(
        str(path)
    ) as document:
        for page_number, page in enumerate(
            document.pages,
            start=1,
        ):
            normal_tables = _extract_valid_tables(
                page,
                allow_text_strategy=False,
            )

            for table_index, table in enumerate(
                normal_tables
            ):
                for row in _normal_table_rows(
                    table
                ):
                    table_chunks.append(
                        _row_record_to_chunk(
                            row,
                            page_number=page_number,
                            table_index=table_index,
                            rotation_applied=0,
                            extraction_mode=(
                                "normal_table_row"
                            ),
                        )
                    )

            if (
                not normal_tables
                and _page_contains_rotated_text(
                    page
                )
            ):
                (
                    coordinate_rows,
                    rotation_used,
                ) = _extract_rotated_coordinate_rows(
                    path=path,
                    page_number=page_number,
                )

                for row in coordinate_rows:
                    table_chunks.append(
                        _row_record_to_chunk(
                            row,
                            page_number=page_number,
                            table_index=0,
                            rotation_applied=(
                                rotation_used or 0
                            ),
                            extraction_mode=(
                                "rotated_coordinate_row"
                            ),
                        )
                    )

    if not table_chunks:
        raise ValueError(
            f"{path.name} produced no coordinate-reconstructed "
            "table rows with pdfplumber"
        )

    vectors: list[list[float]] = []
    batch_size = 32

    for start in range(
        0,
        len(table_chunks),
        batch_size,
    ):
        batch_chunks = table_chunks[
            start : start + batch_size
        ]

        vectors.extend(
            embedder.embed(
                [
                    str(chunk["text"])
                    for chunk in batch_chunks
                ],
                is_query=False,
            )
        )

    # Version 2 malformed table data is replaced automatically.
    table_store.ensure_collection(
        dim=len(vectors[0]),
        reset=(
            reset
            or table_store.exists()
        ),
    )

    points = [
        models.PointStruct(
            id=str(
                uuid.uuid5(
                    _TABLE_NAMESPACE,
                    (
                        f"{path.name}:"
                        f"{chunk['page']}:"
                        f"{chunk['table_index']}:"
                        f"{chunk['row_id']}:"
                        f"{chunk['rotation_applied']}:"
                        f"{_TABLE_INDEX_VERSION}"
                    ),
                )
            ),
            vector=vector,
            payload={
                "text": chunk["text"],
                "raw_row_text": chunk[
                    "raw_row_text"
                ],
                "row_label": chunk[
                    "row_label"
                ],
                "row_id": chunk["row_id"],
                "cells": chunk["cells"],
                "keywords": chunk[
                    "keywords"
                ],
                "statistics": chunk[
                    "statistics"
                ],
                "page": chunk["page"],
                "source": path.name,
                "content_type": "table_row",
                "extractor": (
                    "pdfplumber_coordinates"
                ),
                "table_index": chunk[
                    "table_index"
                ],
                "rotation_applied": chunk[
                    "rotation_applied"
                ],
                "extraction_mode": chunk[
                    "extraction_mode"
                ],
                "index_version": (
                    _TABLE_INDEX_VERSION
                ),
            },
        )
        for chunk, vector in zip(
            table_chunks,
            vectors,
        )
    ]

    table_store.upsert(points)