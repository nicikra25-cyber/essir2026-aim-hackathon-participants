"""Compare PDF extraction quality using three extractors.

The script:
1. extracts selected pages;
2. saves the complete extracted text;
3. checks several indicators of extraction quality;
4. produces a Markdown report with examples and preliminary reasons.
"""

from __future__ import annotations

import re
from pathlib import Path

import pdfplumber
import pymupdf
from pypdf import PdfReader


# =============================================================================
# Settings
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent
PDF_PATH = PROJECT_DIR / "data" / "in" / "Paper_used.pdf"
OUTPUT_DIR = PROJECT_DIR / "extraction_tests"

# PDF page numbers to inspect manually.
PAGES_TO_TEST = [2, 9, 16, 25, 27]

# Important phrases connected to your Level-1 questions.
# Change these if the exact wording in the PDF is different.
TARGET_PHRASES = [
    "threat of new entrants",
    "organisational performance and competitive advantage",
    "505 complete responses were available for data analysis",
]

SNIPPET_LENGTH = 700


# =============================================================================
# General helpers
# =============================================================================

def prepare_output_folder() -> None:
    """Check the PDF and create the output directory."""
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    OUTPUT_DIR.mkdir(exist_ok=True)


def clean_for_comparison(text: str) -> str:
    """Normalise spacing and case for phrase comparisons."""
    text = text.replace("\u00ad", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def save_raw_output(
    extractor_name: str,
    extracted_pages: dict[int, str],
) -> None:
    """Save the complete extracted text for one extractor."""
    output_path = OUTPUT_DIR / f"{extractor_name}_output.txt"

    with output_path.open("w", encoding="utf-8") as output_file:
        for page_number, text in extracted_pages.items():
            output_file.write(
                f"\n{'=' * 80}\n"
                f"PDF PAGE {page_number}\n"
                f"{'=' * 80}\n\n"
                f"{text.strip() or '[NO TEXT EXTRACTED]'}\n"
            )


def extraction_metrics(text: str) -> dict[str, int]:
    """Calculate simple indicators of possible extraction problems."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    return {
        "characters": len(text),
        "lines": len(lines),

        # Example: "organisa-\ntion"
        "hyphenated_line_breaks": len(
            re.findall(r"[A-Za-z]-\s*\n\s*[A-Za-z]", text)
        ),

        # Unicode replacement character often indicates damaged text.
        "replacement_symbols": text.count("�"),

        # Many very short lines may indicate broken columns or tables.
        "very_short_lines": sum(len(line) < 15 for line in lines),
    }


def phrase_matches(text: str) -> list[str]:
    """Return important phrases preserved in the extracted text."""
    normalised_text = clean_for_comparison(text)

    return [
        phrase
        for phrase in TARGET_PHRASES
        if clean_for_comparison(phrase) in normalised_text
    ]


def create_snippet(text: str, length: int = SNIPPET_LENGTH) -> str:
    """Create a readable example from the beginning of the extracted page."""
    cleaned = re.sub(r"\s+", " ", text).strip()

    if not cleaned:
        return "[NO TEXT EXTRACTED]"

    if len(cleaned) <= length:
        return cleaned

    return cleaned[:length] + " ..."


def create_phrase_example(text: str, phrase: str) -> str | None:
    """Show the context surrounding an important phrase."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    normalised = cleaned.lower()
    target = clean_for_comparison(phrase)

    position = normalised.find(target)

    if position == -1:
        return None

    start = max(0, position - 180)
    end = min(len(cleaned), position + len(target) + 180)

    return cleaned[start:end]


# =============================================================================
# Extractors
# =============================================================================

def extract_with_pypdf() -> dict[int, str]:
    """Extract selected pages with pypdf."""
    reader = PdfReader(PDF_PATH)
    extracted_pages: dict[int, str] = {}

    for page_number in PAGES_TO_TEST:
        page = reader.pages[page_number - 1]
        extracted_pages[page_number] = page.extract_text() or ""

    return extracted_pages


def extract_with_pymupdf() -> dict[int, str]:
    """Extract selected pages with PyMuPDF."""
    document = pymupdf.open(PDF_PATH)
    extracted_pages: dict[int, str] = {}

    try:
        for page_number in PAGES_TO_TEST:
            page = document[page_number - 1]

            # sort=True attempts to preserve a natural reading order.
            extracted_pages[page_number] = page.get_text(
                "text",
                sort=True,
            )
    finally:
        document.close()

    return extracted_pages


def extract_with_pdfplumber() -> tuple[dict[int, str], dict[int, int]]:
    """Extract selected pages and count detected tables with pdfplumber."""
    extracted_pages: dict[int, str] = {}
    table_counts: dict[int, int] = {}

    with pdfplumber.open(PDF_PATH) as document:
        for page_number in PAGES_TO_TEST:
            page = document.pages[page_number - 1]

            extracted_pages[page_number] = page.extract_text() or ""
            table_counts[page_number] = len(page.extract_tables())

    return extracted_pages, table_counts


# =============================================================================
# Report
# =============================================================================

def total_metrics(pages: dict[int, str]) -> dict[str, int]:
    """Combine extraction metrics across all inspected pages."""
    totals = {
        "characters": 0,
        "lines": 0,
        "hyphenated_line_breaks": 0,
        "replacement_symbols": 0,
        "very_short_lines": 0,
        "target_phrases": 0,
    }

    for text in pages.values():
        page_metrics = extraction_metrics(text)

        for name, value in page_metrics.items():
            totals[name] += value

        totals["target_phrases"] += len(phrase_matches(text))

    return totals


def preliminary_reason(
    metrics: dict[str, int],
    table_count: int,
) -> str:
    """Produce a cautious preliminary interpretation."""
    reasons: list[str] = []

    if metrics["target_phrases"] == len(TARGET_PHRASES):
        reasons.append("preserved all tested Level-1 phrases")
    else:
        reasons.append(
            f"preserved {metrics['target_phrases']} of "
            f"{len(TARGET_PHRASES)} tested phrases"
        )

    if metrics["hyphenated_line_breaks"] == 0:
        reasons.append("no broken hyphenated words detected")
    else:
        reasons.append(
            f"{metrics['hyphenated_line_breaks']} possible broken words"
        )

    if metrics["replacement_symbols"] > 0:
        reasons.append(
            f"{metrics['replacement_symbols']} damaged-character symbols"
        )

    if table_count > 0:
        reasons.append(f"detected {table_count} tables")

    return "; ".join(reasons)


def create_report(
    results: dict[str, dict[int, str]],
    pdfplumber_table_counts: dict[int, int],
) -> None:
    """Create the comparison report with examples and reasons."""
    report_path = OUTPUT_DIR / "comparison_report.md"

    totals = {
        extractor: total_metrics(pages)
        for extractor, pages in results.items()
    }

    total_tables = sum(pdfplumber_table_counts.values())

    with report_path.open("w", encoding="utf-8") as report:
        report.write("# PDF Extraction Comparison\n\n")

        report.write(
            "This report compares extraction quality on PDF pages "
            f"{', '.join(map(str, PAGES_TO_TEST))}. "
            "The automatic indicators are preliminary and must be checked "
            "against the visible PDF.\n\n"
        )

        report.write("## Summary\n\n")

        report.write(
            "| Extractor | Important phrases found | Broken-word indicators "
            "| Damaged symbols | Tables detected | Preliminary reason |\n"
        )
        report.write(
            "|---|---:|---:|---:|---:|---|\n"
        )

        for extractor, metrics in totals.items():
            tables = total_tables if extractor == "pdfplumber" else 0

            reason = preliminary_reason(metrics, tables)

            report.write(
                f"| {extractor} "
                f"| {metrics['target_phrases']}/{len(TARGET_PHRASES)} "
                f"| {metrics['hyphenated_line_breaks']} "
                f"| {metrics['replacement_symbols']} "
                f"| {tables} "
                f"| {reason} |\n"
            )

        report.write("\n## Important phrase checks\n\n")

        for phrase in TARGET_PHRASES:
            report.write(f"### Target phrase: `{phrase}`\n\n")

            for extractor, pages in results.items():
                found_pages: list[int] = []

                for page_number, text in pages.items():
                    if clean_for_comparison(phrase) in clean_for_comparison(text):
                        found_pages.append(page_number)

                if found_pages:
                    page_text = pages[found_pages[0]]
                    example = create_phrase_example(page_text, phrase)

                    report.write(
                        f"**{extractor}: found on PDF page(s) "
                        f"{found_pages}.**\n\n"
                        f"> {example}\n\n"
                    )
                else:
                    report.write(
                        f"**{extractor}: phrase not found exactly.**\n\n"
                    )

        report.write("\n## Side-by-side page examples\n\n")

        for page_number in PAGES_TO_TEST:
            report.write(f"### PDF page {page_number}\n\n")

            for extractor, pages in results.items():
                text = pages[page_number]
                metrics = extraction_metrics(text)

                report.write(f"#### {extractor}\n\n")
                report.write(
                    f"- Characters: {metrics['characters']}\n"
                    f"- Possible broken words: "
                    f"{metrics['hyphenated_line_breaks']}\n"
                    f"- Very short lines: {metrics['very_short_lines']}\n"
                    f"- Important phrases found: "
                    f"{len(phrase_matches(text))}\n\n"
                )

                report.write("Example:\n\n")
                report.write(f"> {create_snippet(text)}\n\n")

            tables = pdfplumber_table_counts.get(page_number, 0)
            report.write(
                f"**pdfplumber tables detected on this page:** {tables}\n\n"
            )

        report.write("## Manual observations\n\n")

        report.write(
            "Complete these points after comparing the examples with the PDF:\n\n"
            "- Were the columns mixed?\n"
            "- Were sentences placed in the correct order?\n"
            "- Were words broken across lines?\n"
            "- Were headings and footnotes mixed with the main text?\n"
            "- Were tables readable?\n"
            "- Which extractor preserved the Level-1 evidence most exactly?\n\n"
        )

        report.write("## Final decision\n\n")

        report.write(
            "**Selected extractor:** \n\n"
            "**Why it was selected:** \n\n"
            "**Example showing the improvement:** \n"
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    prepare_output_folder()

    print("Extracting pages with pypdf...")
    pypdf_pages = extract_with_pypdf()

    print("Extracting pages with PyMuPDF...")
    pymupdf_pages = extract_with_pymupdf()

    print("Extracting pages with pdfplumber...")
    pdfplumber_pages, table_counts = extract_with_pdfplumber()

    results = {
        "pypdf": pypdf_pages,
        "PyMuPDF": pymupdf_pages,
        "pdfplumber": pdfplumber_pages,
    }

    for extractor, pages in results.items():
        save_raw_output(extractor.lower(), pages)

    create_report(results, table_counts)

    print("\nComparison completed.")
    print(f"Open this report:\n{OUTPUT_DIR / 'comparison_report.md'}")


if __name__ == "__main__":
    main()