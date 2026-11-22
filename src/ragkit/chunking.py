"""Splitting a document into retrievable chunks.

Two decisions worth stating, because both are visible in retrieval quality:

**Chunking spans pages, rather than restarting at each one.** Chunking per page is simpler and
keeps provenance trivially exact, but it fragments every sentence that crosses a page break
and produces a useless 40-character chunk for every short page. So the pages are joined, the
join is chunked, and each chunk's page is recovered from the offset where it starts.

**Boundaries snap backwards to whitespace.** A fixed character cut lands mid-word roughly
every time, and a half-word at each edge is noise in the embedding and an ugly quote in a
citation. Snapping back costs a few characters of chunk size and nothing else.
"""

from __future__ import annotations

from ragkit.extract import ExtractedDocument
from ragkit.models import Chunk, Page

# How far back a boundary may travel to find whitespace. A quarter of the chunk is generous
# enough for prose and bounded enough that a long unbroken run (a base64 blob, a table of
# numbers) falls back to a hard cut instead of collapsing the chunk to nothing.
_SNAP_FRACTION = 4


def chunk_document(
    doc: ExtractedDocument,
    *,
    chunk_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    """Split `doc` into overlapping chunks, each carrying its source page."""
    if overlap_chars >= chunk_chars:
        raise ValueError(
            f"overlap_chars ({overlap_chars}) must be smaller than chunk_chars "
            f"({chunk_chars}); equal or larger cannot advance through the document."
        )

    full_text, page_spans = _join_pages(doc.pages)
    if not full_text:
        return []

    chunks: list[Chunk] = []
    cursor = 0
    ordinal = 0
    length = len(full_text)
    # Guaranteed-positive stride, used only if the overlap arithmetic would stall.
    stride = max(1, chunk_chars - overlap_chars)
    snap_budget = max(1, chunk_chars // _SNAP_FRACTION)

    while cursor < length:
        end = min(length, cursor + chunk_chars)
        if end < length:
            snapped = _snap_back(full_text, cursor, end, snap_budget)
            if snapped > cursor:
                end = snapped

        body = full_text[cursor:end].strip()
        if body:
            page = _page_at(page_spans, cursor) if doc.paginated else None
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{ordinal:04d}",
                    doc_id=doc.doc_id,
                    text=body,
                    ordinal=ordinal,
                    page=page,
                )
            )
            ordinal += 1

        if end >= length:
            break

        next_cursor = end - overlap_chars
        if next_cursor <= cursor:
            # Overlap wider than the snapped chunk. Fall back to the stride so the loop
            # always advances; without this a pathological document hangs the ingest.
            next_cursor = cursor + stride
        cursor = next_cursor

    return chunks


def _join_pages(pages: list[Page]) -> tuple[str, list[tuple[int, int, int]]]:
    """Concatenate pages, recording `(page_number, start, end)` for each in the result."""
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    offset = 0
    separator = "\n\n"

    for page in pages:
        text = page.text.strip()
        if not text:
            continue  # Blank pages contribute no retrievable text and no offsets.
        if parts:
            offset += len(separator)
            parts.append(separator)
        spans.append((page.number, offset, offset + len(text)))
        parts.append(text)
        offset += len(text)

    return "".join(parts), spans


def _page_at(spans: list[tuple[int, int, int]], offset: int) -> int | None:
    """The page containing `offset`.

    A chunk that starts inside the separator between two pages is attributed to the page it
    runs into, which is where its first real word lives.
    """
    if not spans:
        return None
    for number, start, end in spans:
        if start <= offset < end:
            return number
        if offset < start:
            return number
    return spans[-1][0]


def _snap_back(text: str, lower_bound: int, end: int, budget: int) -> int:
    """Move `end` back to the nearest whitespace, within `budget` characters.

    Returns `end` unchanged when there is no whitespace to find, so an unbroken run is cut
    hard rather than producing an empty chunk.
    """
    floor = max(lower_bound + 1, end - budget)
    for i in range(end, floor - 1, -1):
        if i > 0 and text[i - 1].isspace():
            return i
    return end


def chunk_count_estimate(chars: int, *, chunk_chars: int, overlap_chars: int) -> int:
    """Rough chunk count for `chars` of text, for progress reporting and capacity planning."""
    stride = max(1, chunk_chars - overlap_chars)
    if chars <= chunk_chars:
        return 1 if chars else 0
    return 1 + -(-(chars - chunk_chars) // stride)  # ceiling division
