"""End-to-end tests through `RagPipeline`.

The valuable cases here are the ones a real model cannot be made to perform on demand: a
fabricated quote, a citation to a passage that was never supplied, a reply that will not
parse. `ScriptedLLM` supplies those, so the verification layer is tested *through the
pipeline* rather than only in isolation — which is where a wiring mistake would hide.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.conftest import ScriptedLLM, draft_json, make_pipeline

from ragkit.generate import ExtractiveStub
from ragkit.pipeline import RagPipeline


class TestIngest:
    def test_indexes_a_document(self, pipeline: RagPipeline, policy_doc: Path):
        report = pipeline.ingest(policy_doc)
        assert report.doc_id == "notice_policy.md"
        assert report.chunks_written > 1
        assert report.chars_extracted > 0
        assert pipeline.store.count() == report.chunks_written

    def test_reingesting_does_not_duplicate(self, pipeline: RagPipeline, policy_doc: Path):
        first = pipeline.ingest(policy_doc)
        pipeline.ingest(policy_doc)
        assert pipeline.store.count() == first.chunks_written

    def test_two_documents_coexist(self, pipeline: RagPipeline, corpus_dir: Path, pdf_path: Path):
        a = pipeline.ingest(corpus_dir / "notice_policy.md")
        b = pipeline.ingest(pdf_path)
        assert pipeline.store.count() == a.chunks_written + b.chunks_written

    def test_doc_id_flows_into_citations(self, pipeline: RagPipeline, policy_doc: Path):
        pipeline.ingest(policy_doc, doc_id="employee-handbook.md")
        answer = pipeline.ask("How much notice must an employee give before resigning?")
        assert not answer.refused
        assert answer.citations[0].doc_id == "employee-handbook.md"


class TestAskGrounded:
    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("How much notice must an employee give before resigning?", "thirty days"),
            ("How many days of annual leave do employees accrue?", "twenty-one days"),
            ("What notice is required during probation?", "seven days"),
            ("What fee applies to equipment that is not returned?", "400 pounds"),
        ],
    )
    def test_answers_are_grounded_and_correct(self, indexed_pipeline, question, expected):
        answer = indexed_pipeline.ask(question)
        assert not answer.refused, answer.refusal_reason
        assert answer.groundedness.verdict == "grounded"
        assert expected in answer.text.lower()
        assert answer.citations

    def test_every_citation_quote_is_present_in_its_chunk(self, indexed_pipeline):
        from ragkit.ground import normalise_for_match

        answer = indexed_pipeline.ask("How much notice must an employee give before resigning?")
        by_id = {r.chunk.chunk_id: r.chunk for r in answer.retrieved}
        for citation in answer.citations:
            source = by_id[citation.chunk_id]
            assert normalise_for_match(citation.quote) in normalise_for_match(source.text)

    def test_rendered_output_lists_sources(self, indexed_pipeline):
        answer = indexed_pipeline.ask("How many days of annual leave do employees accrue?")
        rendered = answer.rendered()
        assert "[1]" in rendered
        assert "notice_policy.md" in rendered

    def test_report_records_the_numbers(self, indexed_pipeline):
        answer = indexed_pipeline.ask("What notice is required during probation?")
        report = answer.groundedness
        assert report.chunks_retrieved > 0
        assert report.support_verified >= 1
        assert report.best_score >= report.score_threshold


class TestAskRefuses:
    @pytest.mark.parametrize(
        "question",
        [
            "What is the capital of Mongolia?",
            "Who won the 1998 World Cup final?",
            "What is the airspeed velocity of an unladen swallow?",
        ],
    )
    def test_out_of_corpus_questions_are_refused_before_generation(
        self, embedder, store, settings, policy_doc, question
    ):
        # A counting LLM proves the gate short-circuits: no tokens are spent on a question
        # the corpus cannot answer.
        llm = ScriptedLLM(draft_json(sufficient=False))
        pipe = make_pipeline(embedder, store, settings, llm)
        pipe.ingest(policy_doc)

        answer = pipe.ask(question)
        assert answer.refused
        assert answer.groundedness.verdict == "insufficient_retrieval"
        assert llm.calls == [], "the model was called despite failing the retrieval gate"
        assert answer.refusal_reason

    def test_empty_question(self, indexed_pipeline):
        answer = indexed_pipeline.ask("   ")
        assert answer.refused
        assert answer.groundedness.verdict == "insufficient_retrieval"

    def test_empty_index(self, pipeline: RagPipeline):
        answer = pipeline.ask("How much notice is required?")
        assert answer.refused
        assert answer.groundedness.chunks_retrieved == 0
        assert "nothing is indexed" in (answer.refusal_reason or "").lower()

    def test_model_declining_is_respected(self, embedder, store, settings, policy_doc):
        pipe = make_pipeline(embedder, store, settings, ScriptedLLM(draft_json(sufficient=False)))
        pipe.ingest(policy_doc)
        answer = pipe.ask("How much notice must an employee give before resigning?")
        assert answer.refused
        assert answer.groundedness.verdict == "model_declined"


class TestFabricationIsWithheld:
    """The cases that justify the project."""

    def test_fabricated_quote_is_not_published(self, embedder, store, settings, policy_doc):
        llm = ScriptedLLM(
            draft_json(
                answer="Employees must give ninety days notice [1].",
                support=[{"passage": 1, "quote": "Employees must give ninety days notice"}],
            )
        )
        pipe = make_pipeline(embedder, store, settings, llm)
        pipe.ingest(policy_doc)

        answer = pipe.ask("How much notice must an employee give before resigning?")
        assert answer.refused, "a fabricated quote was published"
        assert answer.text is None
        assert answer.citations == []
        assert answer.groundedness.verdict == "ungrounded"
        assert answer.groundedness.support_verified == 0

    def test_citation_to_an_unsupplied_passage_is_not_published(
        self, embedder, store, settings, policy_doc
    ):
        llm = ScriptedLLM(
            draft_json(
                answer="See the appendix [9].",
                support=[{"passage": 9, "quote": "a passage that was never supplied here"}],
            )
        )
        pipe = make_pipeline(embedder, store, settings, llm)
        pipe.ingest(policy_doc)

        answer = pipe.ask("How much notice must an employee give before resigning?")
        assert answer.refused
        assert 9 in answer.groundedness.rejected_out_of_range

    def test_answer_with_no_citations_is_not_published(self, embedder, store, settings, policy_doc):
        llm = ScriptedLLM(draft_json(answer="Thirty days, I think.", support=[]))
        pipe = make_pipeline(embedder, store, settings, llm)
        pipe.ingest(policy_doc)

        answer = pipe.ask("How much notice must an employee give before resigning?")
        assert answer.refused
        assert answer.groundedness.support_claimed == 0

    def test_partially_supported_answer_is_withheld_entirely(
        self, embedder, store, settings, policy_doc
    ):
        # Passage 1 verifies; the second claim cites a passage with no evidence. Publishing
        # the verified half with a dangling citation is the failure mode being prevented.
        llm = ScriptedLLM(
            draft_json(
                answer="Thirty days [1], and unlimited carry-over applies [2].",
                support=[
                    {"passage": 1, "quote": "thirty days written notice"},
                    {"passage": 2, "quote": "unlimited carry-over of annual leave applies"},
                ],
            )
        )
        pipe = make_pipeline(embedder, store, settings, llm)
        pipe.ingest(policy_doc)

        answer = pipe.ask("How much notice must an employee give before resigning?")
        assert answer.refused
        assert answer.text is None


class TestMalformedResponses:
    def test_retries_once_then_refuses(self, embedder, store, settings, policy_doc):
        llm = ScriptedLLM("not json at all")
        pipe = make_pipeline(embedder, store, settings, llm)
        pipe.ingest(policy_doc)

        answer = pipe.ask("How much notice must an employee give before resigning?")
        assert answer.refused
        assert answer.groundedness.verdict == "model_declined"
        assert len(llm.calls) == 2, "expected exactly one retry"

    def test_recovers_when_the_retry_parses(self, embedder, store, settings, policy_doc):
        llm = ScriptedLLM(
            "```\nnot json\n```",
            draft_json(
                answer="Employees must give thirty days written notice [1].",
                support=[{"passage": 1, "quote": "thirty days written notice"}],
            ),
        )
        pipe = make_pipeline(embedder, store, settings, llm)
        pipe.ingest(policy_doc)

        answer = pipe.ask("How much notice must an employee give before resigning?")
        assert not answer.refused, answer.refusal_reason
        assert len(llm.calls) == 2

    def test_a_malformed_response_never_raises(self, embedder, store, settings, policy_doc):
        # From a caller's point of view this is "no grounded answer", not a 500.
        pipe = make_pipeline(embedder, store, settings, ScriptedLLM("{"))
        pipe.ingest(policy_doc)
        assert pipe.ask("How much notice is required?").refused


class TestConstruction:
    def test_from_settings_builds_an_offline_stack(self, settings):
        pipe = RagPipeline.from_settings(settings)
        assert pipe.store.count() == 0
        assert isinstance(pipe.llm, ExtractiveStub)
        assert pipe.embedder.name == "hashing-512"

    def test_invalid_settings_are_rejected(self, settings):
        bad = settings.model_copy(update={"chunk_overlap_chars": 999})
        with pytest.raises(ValueError, match="must be smaller"):
            RagPipeline.from_settings(bad)
