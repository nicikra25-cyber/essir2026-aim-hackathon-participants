"""Turn PDF pages into the units you index.

Each page is split into chunks of three sentences with one sentence of overlap.
Chunks never cross page boundaries, so the correct PDF page is preserved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


SENTENCES_PER_CHUNK = 3
SENTENCE_OVERLAP = 1


@dataclass
class Chunk:
    text: str
    page: int      # 1-indexed
    index: int     # position within the document


def chunk_pages(pages: list[str]) -> list[Chunk]:
    """Split each page into three-sentence chunks with one-sentence overlap.

    Example:
      chunk 1 = sentences 1, 2, 3
      chunk 2 = sentences 3, 4, 5

    Chunks remain inside their original PDF page so citations still line up.

    TODO(level-3): a flat chunk loses where it sits in the document. Section titles,
      or a small/large ("parent") hierarchy, help with whole-document questions.
    """
    chunks: list[Chunk] = []
    idx = 0

    step = SENTENCES_PER_CHUNK - SENTENCE_OVERLAP

    for page_no, text in enumerate(pages, start=1):
        text = text.strip()

        if not text:
            continue

        # Replace line breaks and repeated spaces with one normal space.
        text = re.sub(r"\s+", " ", text)

        # Split the page after sentence-ending punctuation.
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        for start in range(0, len(sentences), step):
            selected_sentences = sentences[
                start : start + SENTENCES_PER_CHUNK
            ]

            if not selected_sentences:
                continue

            chunk_text = " ".join(selected_sentences)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    page=page_no,
                    index=idx,
                )
            )
            idx += 1

            # Stop after the final sentences of the page.
            if start + SENTENCES_PER_CHUNK >= len(sentences):
                break

    return chunks