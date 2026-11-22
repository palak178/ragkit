"""Tests for the deterministic verification layer.

This is the file that matters most. Everything else in the package can be wrong in ways that
degrade quality; `ground.py` being wrong means the project's central claim — that a published
answer's citations have been checked — is false, which is the exact defect this rewrite
exists to remove. So the fabrication cases are tested at least as hard as the happy path.
"""

from __future__ import annotations

import pytest

from ragkit.ground import (
    MIN_QUOTE_CHARS,
    gate_retrieval,
    inline_markers,
    normalise_for_match,
    refusal_text,
    verify,
)
from ragkit.models import Chunk, RetrievedChunk


def chunk(text: str, *, cid: str = "d:0000", page: int | None = 1) -> Chunk:
    return Chunk(chunk_id=cid, doc_id="policy.md", text=text, ordinal=0, page=page)


def retrieved(*texts: str, scores: list[float] | None = None) -> list[RetrievedChunk]:
    scores = scores or [0.9 - 0.1 * i for i in range(len(texts))]
    return [
        RetrievedChunk(chunk=chunk(t, cid=f"d:{i:04d}"), score=s)
        for i, (t, s) in enumerate(zip(texts, scores, strict=True))
    ]


# ---------------------------------------------------------------------------------------
# normalise_for_match
# ---------------------------------------------------------------------------------------


class TestNormalisation:
    def test_collapses_whitespace_and_case(self):
        assert normalise_for_match("  Thirty   DAYS\n notice ") == "thirty days notice"

    @pytest.mark.parametrize(
        ("fancy", "plain"),
        [
            ("don’t", "don't"),
            ("“quoted”", '"quoted"'),
            ("thirty–day", "thirty-day"),
            ("a b", "a b"),
            ("wait…", "wait..."),
        ],
    )
    def test_folds_punctuation_a_model_would_substitute(self, fancy, plain):
        # A model that retypes a quote with curly quotes has not fabricated anything, and
        # must not be reported as if it had.
        assert normalise_for_match(fancy) == normalise_for_match(plain)

    def test_nfkc_folds_compatibility_forms(self):
        assert normalise_for_match("ﬁle") == normalise_for_match("file")

    def test_does_not_reorder_or_drop_words(self):
        # The safety property: normalisation may only make a true match findable, never
        # invent one. If it dropped words, "A then B" would match "B then A".
        assert normalise_for_match("alpha beta") != normalise_for_match("beta alpha")


class TestInlineMarkers:
    def test_extracts_in_order_without_duplicates(self):
        assert inline_markers("First [2] then [1] and again [2].") == [2, 1]

    def test_ignores_non_numeric_brackets(self):
        assert inline_markers("as written [sic] and [see below]") == []

    def test_no_markers(self):
        assert inline_markers("no citations here") == []


# ---------------------------------------------------------------------------------------
# Gate 1: retrieval
# ---------------------------------------------------------------------------------------


class TestRetrievalGate:
    def test_nothing_retrieved_is_refused(self):
        report = gate_retrieval([], threshold=0.2)
        assert report is not None
        assert report.verdict == "insufficient_retrieval"
        assert report.chunks_retrieved == 0
        assert report.best_score == -1.0

    def test_below_threshold_is_refused(self):
        report = gate_retrieval(retrieved("irrelevant text", scores=[0.05]), threshold=0.2)
        assert report is not None
        assert report.verdict == "insufficient_retrieval"
        assert report.best_score == pytest.approx(0.05)

    def test_at_threshold_proceeds(self):
        # Boundary is inclusive; documented as "minimum top-1 similarity".
        assert gate_retrieval(retrieved("text", scores=[0.2]), threshold=0.2) is None

    def test_above_threshold_proceeds(self):
        assert gate_retrieval(retrieved("text", scores=[0.9]), threshold=0.2) is None


# ---------------------------------------------------------------------------------------
# Gate 2: verification
# ---------------------------------------------------------------------------------------

PASSAGE = "Employees must give thirty days written notice before resigning from the company."


class TestVerify:
    def test_verbatim_quote_is_grounded(self):
        report, citations = verify(
            answer="Thirty days [1].",
            support=[{"passage": 1, "quote": "thirty days written notice"}],
            retrieved=retrieved(PASSAGE),
            threshold=0.2,
        )
        assert report.verdict == "grounded"
        assert report.support_verified == 1
        assert [c.marker for c in citations] == [1]
        assert citations[0].chunk_id == "d:0000"
        assert citations[0].page == 1

    def test_fabricated_quote_is_rejected(self):
        # The headline failure mode: a plausible sentence that is simply not in the source.
        report, citations = verify(
            answer="Ninety days [1].",
            support=[{"passage": 1, "quote": "ninety days written notice is required"}],
            retrieved=retrieved(PASSAGE),
            threshold=0.2,
        )
        assert report.verdict == "ungrounded"
        assert report.support_verified == 0
        assert report.rejected_quote_not_found == [1]
        assert citations == []

    def test_citing_a_passage_that_was_never_supplied(self):
        report, citations = verify(
            answer="See [7].",
            support=[{"passage": 7, "quote": "something from nowhere"}],
            retrieved=retrieved(PASSAGE),
            threshold=0.2,
        )
        assert report.verdict == "ungrounded"
        assert 7 in report.rejected_out_of_range
        assert citations == []

    def test_inline_marker_with_no_supporting_quote_refuses_everything(self):
        # Strictness is deliberate: passage 1 verified, but the claim attributed to passage 2
        # has no evidence, so the whole answer is withheld rather than published with a hole.
        report, citations = verify(
            answer="Thirty days [1], and equipment fees apply [2].",
            support=[{"passage": 1, "quote": "thirty days written notice"}],
            retrieved=retrieved(PASSAGE, "Unrelated second passage about parking permits."),
            threshold=0.2,
        )
        assert report.verdict == "ungrounded"
        assert report.support_verified == 1
        assert 2 in report.rejected_out_of_range
        assert citations == []

    def test_quote_shorter_than_the_minimum_is_rejected(self):
        short = PASSAGE.split()[0][:3]
        assert len(short) < MIN_QUOTE_CHARS
        report, _ = verify(
            answer="Yes [1].",
            support=[{"passage": 1, "quote": short}],
            retrieved=retrieved(PASSAGE),
            threshold=0.2,
        )
        assert report.verdict == "ungrounded"
        assert report.rejected_quote_not_found == [1]

    def test_no_support_offered_is_ungrounded(self):
        report, _ = verify(
            answer="Thirty days.", support=[], retrieved=retrieved(PASSAGE), threshold=0.2
        )
        assert report.verdict == "ungrounded"
        assert report.support_claimed == 0

    def test_empty_answer_with_valid_support_is_still_ungrounded(self):
        report, _ = verify(
            answer="   ",
            support=[{"passage": 1, "quote": "thirty days written notice"}],
            retrieved=retrieved(PASSAGE),
            threshold=0.2,
        )
        assert report.verdict == "ungrounded"

    def test_quote_requoted_with_different_punctuation_still_verifies(self):
        source = "Unused leave expires on 31 March — it is not paid out."
        report, citations = verify(
            answer="It expires [1].",
            support=[{"passage": 1, "quote": "expires on 31 March - it is not paid out"}],
            retrieved=retrieved(source),
            threshold=0.2,
        )
        assert report.verdict == "grounded"
        assert len(citations) == 1

    def test_duplicate_support_for_one_passage_yields_one_citation(self):
        report, citations = verify(
            answer="Thirty days [1].",
            support=[
                {"passage": 1, "quote": "thirty days written notice"},
                {"passage": 1, "quote": "before resigning from the company"},
            ],
            retrieved=retrieved(PASSAGE),
            threshold=0.2,
        )
        assert report.verdict == "grounded"
        assert len(citations) == 1

    @pytest.mark.parametrize(
        "entry",
        [
            {},
            {"passage": "one", "quote": "thirty days written notice"},
            {"passage": None, "quote": "thirty days written notice"},
            {"passage": 1},
            {"passage": 1, "quote": None},
        ],
    )
    def test_malformed_support_entries_do_not_crash(self, entry):
        # These arrive straight from a model, so the shapes are not hypothetical.
        report, citations = verify(
            answer="Something [1].",
            support=[entry],
            retrieved=retrieved(PASSAGE),
            threshold=0.2,
        )
        assert report.verdict == "ungrounded"
        assert citations == []

    def test_citations_ordered_by_first_appearance_in_the_answer(self):
        two = "Company laptops must be returned on the final working day."
        report, citations = verify(
            answer="Return laptops [2], and give notice [1].",
            support=[
                {"passage": 1, "quote": "thirty days written notice"},
                {"passage": 2, "quote": "returned on the final working day"},
            ],
            retrieved=retrieved(PASSAGE, two),
            threshold=0.2,
        )
        assert report.verdict == "grounded"
        assert [c.marker for c in citations] == [2, 1]


class TestRefusalText:
    def test_empty_index_message_is_actionable(self):
        report = gate_retrieval([], threshold=0.2)
        assert report is not None
        assert "nothing is indexed" in refusal_text(report).lower()

    def test_below_threshold_message_quotes_the_numbers(self):
        report = gate_retrieval(retrieved("x", scores=[0.05]), threshold=0.2)
        assert report is not None
        text = refusal_text(report)
        assert "0.050" in text and "0.200" in text

    def test_ungrounded_message_names_the_failure(self):
        report, _ = verify(
            answer="Ninety days [1].",
            support=[{"passage": 1, "quote": "ninety days of written notice required"}],
            retrieved=retrieved(PASSAGE),
            threshold=0.2,
        )
        assert "failed verification" in refusal_text(report)

    def test_grounded_report_has_no_refusal_text(self):
        report, _ = verify(
            answer="Thirty days [1].",
            support=[{"passage": 1, "quote": "thirty days written notice"}],
            retrieved=retrieved(PASSAGE),
            threshold=0.2,
        )
        assert refusal_text(report) == ""
