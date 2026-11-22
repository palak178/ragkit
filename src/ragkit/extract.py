"""Getting text out of files.

A PDF is a compressed binary container, so decoding its bytes with a text codec does not
yield text — it yields mojibake that embeds cleanly, indexes cleanly and poisons every claim
downstream. This module uses a real parser, and refuses files it cannot actually read rather
than passing through whatever bytes it managed to decode.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from ragkit.errors import ExtractionError, UnsupportedFormatError
from ragkit.models import Page

# Lower-cased suffixes this module can handle. Checked against the real extension rather than
# a user-supplied content type, which is trivially wrong or spoofed.
PDF_SUFFIXES = {".pdf"}
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}
SUPPORTED_SUFFIXES = PDF_SUFFIXES | TEXT_SUFFIXES


class ExtractedDocument(BaseModel):
    """Text pulled from one file, before chunking."""

    doc_id: str
    pages: list[Page]
    paginated: bool
    """True when page numbers are real coordinates in the source.

    Plain text has no pages. Reporting `p.1` for a 40 000-word Markdown file would be a
    citation that looks precise and is not, so non-paginated formats carry `None` through to
    the chunk and the citation simply names the document.
    """

    @property
    def chars(self) -> int:
        return sum(len(p.text) for p in self.pages)


def extract(path: str | Path, *, doc_id: str | None = None) -> ExtractedDocument:
    """Read `path` into pages of text.

    `doc_id` defaults to the file name, which is what a citation should show. Pass it
    explicitly when ingesting from a temporary upload path, so the citation names the file the
    user recognises rather than `tmpv0c1ab`.
    """
    p = Path(path)
    if not p.is_file():
        raise ExtractionError(f"Not a file: {p}")

    suffix = p.suffix.lower()
    identifier = doc_id or p.name

    if suffix in PDF_SUFFIXES:
        doc = _extract_pdf(p, identifier)
    elif suffix in TEXT_SUFFIXES:
        doc = _extract_text(p, identifier)
    else:
        raise UnsupportedFormatError(
            f"No extractor for '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    if doc.chars == 0:
        # A scanned PDF parses fine and yields nothing, because the text is pixels. Failing
        # loudly here is the difference between "this needs OCR" and an index that silently
        # contains nothing while every later stage reports success.
        raise ExtractionError(
            f"{identifier} produced no text. A scanned or image-only PDF needs OCR, which this "
            f"package does not do."
        )
    return doc


def _extract_pdf(path: Path, doc_id: str) -> ExtractedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - pypdf is a core dependency
        raise ExtractionError("pypdf is required to read PDFs") from exc

    try:
        reader = PdfReader(str(path))
        pages = [
            Page(number=i, text=_normalise(page.extract_text() or ""))
            for i, page in enumerate(reader.pages, start=1)
        ]
    except Exception as exc:
        # pypdf raises a wide and version-dependent set of exceptions on damaged files.
        # Narrowing this would mean guessing at them; wrapping keeps one error type at the
        # boundary while preserving the original as `__cause__`.
        raise ExtractionError(f"Could not read {doc_id}: {exc}") from exc

    return ExtractedDocument(doc_id=doc_id, pages=pages, paginated=True)


def _extract_text(path: Path, doc_id: str) -> ExtractedDocument:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ExtractionError(
            f"{doc_id} is not valid UTF-8. Convert it first rather than letting a lossy "
            f"fallback encoding corrupt the text silently."
        ) from exc
    return ExtractedDocument(
        doc_id=doc_id,
        pages=[Page(number=1, text=_normalise(raw))],
        paginated=False,
    )


def _normalise(text: str) -> str:
    """Collapse the whitespace noise that PDF extraction introduces.

    Kept conservative on purpose: runs of spaces and tabs collapse, blank lines survive as
    paragraph breaks. Quote verification later checks that a model's quote appears in the
    chunk, so any normalisation applied here must also be applied there — see
    `ground.normalise_for_match`.
    """
    lines = [" ".join(line.split()) for line in text.splitlines()]
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()
