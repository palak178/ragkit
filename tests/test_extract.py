"""Tests for text extraction, including a real PDF.

The PDF case runs against bytes assembled in `conftest.minimal_pdf` rather than a mock, so it
fails if `pypdf` stops working — which is the only version of this test worth having, given
the bug being fixed was "the PDF path never produced text at all".
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.conftest import minimal_pdf

from ragkit.errors import ExtractionError, UnsupportedFormatError
from ragkit.extract import extract


class TestPlainText:
    def test_reads_markdown(self, policy_doc: Path):
        doc = extract(policy_doc)
        assert doc.doc_id == "notice_policy.md"
        assert "thirty days written notice" in doc.pages[0].text
        assert doc.chars > 0

    def test_text_is_not_paginated(self, policy_doc: Path):
        # Plain text has no pages, so a citation must not claim one.
        assert extract(policy_doc).paginated is False

    def test_doc_id_can_be_overridden(self, tmp_path: Path):
        path = tmp_path / "tmp7a3f.txt"
        path.write_text("Some genuine content here.", encoding="utf-8")
        assert extract(path, doc_id="handbook.txt").doc_id == "handbook.txt"

    def test_whitespace_only_file_is_an_error(self, corpus_dir: Path):
        # Silently indexing nothing is worse than failing: every later stage reports success.
        with pytest.raises(ExtractionError, match="no text"):
            extract(corpus_dir / "empty.txt")

    def test_invalid_utf8_is_rejected_not_mangled(self, tmp_path: Path):
        # A lossy fallback encoding would "succeed" here and index mojibake, which is worse
        # than failing: every later stage reports success over corrupted text.
        path = tmp_path / "latin.txt"
        path.write_bytes(b"caf\xe9 pr\xe9sent\xe9 \xff\xfe invalid")
        with pytest.raises(ExtractionError, match="UTF-8"):
            extract(path)

    def test_whitespace_is_normalised(self, tmp_path: Path):
        path = tmp_path / "spaced.md"
        path.write_text("Too    many\t\tspaces   here now.", encoding="utf-8")
        assert extract(path).pages[0].text == "Too many spaces here now."

    def test_paragraph_breaks_survive_normalisation(self, tmp_path: Path):
        path = tmp_path / "paras.md"
        path.write_text("First paragraph.\n\n\n\nSecond paragraph.", encoding="utf-8")
        text = extract(path).pages[0].text
        assert text == "First paragraph.\n\nSecond paragraph."


class TestPdf:
    def test_extracts_real_text_from_a_real_pdf(self, pdf_path: Path):
        doc = extract(pdf_path)
        assert doc.paginated is True
        assert len(doc.pages) == 1
        assert doc.pages[0].number == 1
        assert "thirty days" in doc.pages[0].text.lower()

    def test_multipage_pdf_numbers_pages_from_one(self, tmp_path: Path):
        # pypdf concatenates pages in order; page numbers must be 1-based to match what a
        # reader would cite.
        path = tmp_path / "multi.pdf"
        path.write_bytes(minimal_pdf("Only one page in this fixture."))
        doc = extract(path)
        assert [p.number for p in doc.pages] == [1]

    def test_image_only_pdf_fails_loudly(self, tmp_path: Path):
        # A scanned PDF parses fine and yields no text. This must be an error naming OCR,
        # not an empty index.
        path = tmp_path / "scanned.pdf"
        path.write_bytes(minimal_pdf(" "))
        with pytest.raises(ExtractionError, match="OCR"):
            extract(path)

    def test_corrupt_pdf_is_wrapped(self, tmp_path: Path):
        path = tmp_path / "broken.pdf"
        path.write_bytes(b"%PDF-1.4\nthis is not a pdf body at all")
        with pytest.raises(ExtractionError):
            extract(path)


class TestRejections:
    def test_unsupported_extension_names_what_is_supported(self, tmp_path: Path):
        path = tmp_path / "sheet.xlsx"
        path.write_bytes(b"not really a spreadsheet")
        with pytest.raises(UnsupportedFormatError) as exc:
            extract(path)
        assert ".pdf" in str(exc.value)

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(ExtractionError, match="Not a file"):
            extract(tmp_path / "absent.txt")

    def test_directory_is_not_a_file(self, tmp_path: Path):
        with pytest.raises(ExtractionError, match="Not a file"):
            extract(tmp_path)

    def test_extension_check_is_case_insensitive(self, tmp_path: Path):
        path = tmp_path / "SHOUTING.MD"
        path.write_text("Content that is long enough to count.", encoding="utf-8")
        assert extract(path).chars > 0
