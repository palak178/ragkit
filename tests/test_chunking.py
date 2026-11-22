"""Tests for chunking: overlap, page provenance, and the ways a loop can fail to advance."""

from __future__ import annotations

import pytest

from ragkit.chunking import chunk_count_estimate, chunk_document
from ragkit.extract import ExtractedDocument
from ragkit.models import Page


def doc(*page_texts: str, paginated: bool = True, doc_id: str = "d.pdf") -> ExtractedDocument:
    return ExtractedDocument(
        doc_id=doc_id,
        pages=[Page(number=i, text=t) for i, t in enumerate(page_texts, start=1)],
        paginated=paginated,
    )


class TestBasics:
    def test_short_document_is_one_chunk(self):
        chunks = chunk_document(doc("Short text."), chunk_chars=500, overlap_chars=50)
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."
        assert chunks[0].ordinal == 0

    def test_empty_document_yields_nothing(self):
        assert chunk_document(doc(""), chunk_chars=500, overlap_chars=50) == []

    def test_chunk_ids_are_stable_and_ordered(self):
        text = " ".join(f"word{i}" for i in range(400))
        chunks = chunk_document(doc(text), chunk_chars=300, overlap_chars=50)
        assert len(chunks) > 1
        assert [c.chunk_id for c in chunks] == [f"d.pdf:{i:04d}" for i in range(len(chunks))]
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))

    def test_reingesting_produces_identical_ids(self):
        # This is what makes upsert idempotent rather than duplicating a document.
        text = " ".join(f"word{i}" for i in range(200))
        first = chunk_document(doc(text), chunk_chars=300, overlap_chars=50)
        second = chunk_document(doc(text), chunk_chars=300, overlap_chars=50)
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
        assert [c.text for c in first] == [c.text for c in second]


class TestCoverageAndOverlap:
    def test_every_word_survives_chunking(self):
        words = [f"w{i}" for i in range(500)]
        chunks = chunk_document(doc(" ".join(words)), chunk_chars=250, overlap_chars=40)
        joined = " ".join(c.text for c in chunks)
        missing = [w for w in words if w not in joined]
        assert missing == [], f"chunking lost {len(missing)} words"

    def test_neighbouring_chunks_share_text(self):
        chunks = chunk_document(
            doc(" ".join(f"w{i}" for i in range(400))), chunk_chars=300, overlap_chars=80
        )
        assert len(chunks) >= 2
        # The tail of one chunk should reappear at the head of the next; that is the whole
        # point of overlap, so a fact spanning a boundary survives somewhere intact.
        tail_words = chunks[0].text.split()[-3:]
        assert any(w in chunks[1].text for w in tail_words)

    def test_boundaries_do_not_split_words(self):
        chunks = chunk_document(
            doc(" ".join(f"word{i:03d}" for i in range(300))), chunk_chars=260, overlap_chars=40
        )
        for c in chunks:
            # Every token is `wordNNN`; a mid-word cut would leave a fragment like `wor`.
            for token in c.text.split():
                assert token.startswith("word") and len(token) == 7, f"split token {token!r}"


class TestPageProvenance:
    def test_page_recorded_for_paginated_documents(self):
        chunks = chunk_document(
            doc("alpha " * 60, "beta " * 60, "gamma " * 60), chunk_chars=200, overlap_chars=20
        )
        pages = {c.page for c in chunks}
        assert pages <= {1, 2, 3}
        assert None not in pages
        assert chunks[0].page == 1
        assert chunks[-1].page == 3

    def test_non_paginated_documents_have_no_page(self):
        chunks = chunk_document(
            doc("text " * 200, paginated=False), chunk_chars=200, overlap_chars=20
        )
        assert all(c.page is None for c in chunks)
        assert chunks[0].label == "d.pdf"

    def test_blank_pages_do_not_shift_page_numbers(self):
        # A blank page contributes no text; attributing later text to it would make every
        # subsequent citation point at the wrong page.
        chunks = chunk_document(
            doc("alpha " * 50, "", "gamma " * 50), chunk_chars=180, overlap_chars=20
        )
        assert 2 not in {c.page for c in chunks}
        assert {c.page for c in chunks} == {1, 3}

    def test_label_includes_page(self):
        chunks = chunk_document(doc("alpha " * 50), chunk_chars=500, overlap_chars=20)
        assert chunks[0].label == "d.pdf p.1"


class TestGuards:
    def test_overlap_equal_to_size_is_rejected(self):
        with pytest.raises(ValueError, match="must be smaller"):
            chunk_document(doc("text " * 100), chunk_chars=200, overlap_chars=200)

    def test_overlap_larger_than_size_is_rejected(self):
        with pytest.raises(ValueError, match="must be smaller"):
            chunk_document(doc("text " * 100), chunk_chars=200, overlap_chars=500)

    def test_unbroken_run_terminates(self):
        # No whitespace to snap to anywhere. The snap-back must fall through to a hard cut
        # rather than returning the same boundary forever.
        chunks = chunk_document(doc("x" * 2000), chunk_chars=300, overlap_chars=100)
        assert len(chunks) > 1
        assert all(c.text for c in chunks)

    def test_large_overlap_with_short_snap_still_advances(self):
        chunks = chunk_document(doc("ab " * 1000), chunk_chars=120, overlap_chars=115)
        assert len(chunks) > 1
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))


class TestEstimate:
    @pytest.mark.parametrize(
        ("chars", "expected"),
        [(0, 0), (100, 1), (500, 1), (501, 2)],
    )
    def test_estimate_edges(self, chars, expected):
        assert chunk_count_estimate(chars, chunk_chars=500, overlap_chars=100) == expected

    def test_estimate_is_in_the_right_region(self):
        text = " ".join(f"w{i}" for i in range(2000))
        actual = len(chunk_document(doc(text), chunk_chars=400, overlap_chars=80))
        estimate = chunk_count_estimate(len(text), chunk_chars=400, overlap_chars=80)
        # Snapping makes chunks slightly smaller than the target, so the estimate is a lower
        # bound rather than exact. Assert the relationship, not a magic number.
        assert estimate <= actual <= estimate + 3
