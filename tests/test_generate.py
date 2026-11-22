"""Tests for prompt construction, response parsing and the offline stub.

`parse_draft` is the boundary where an untrusted string becomes structured data, so the
parametrised rejection cases below are the real content of this file. The rule it enforces:
tolerate formatting habits, never guess at meaning.
"""

from __future__ import annotations

import json

import pytest

from ragkit.errors import GenerationError, MalformedLLMResponseError
from ragkit.generate import (
    AnthropicMessages,
    ExtractiveStub,
    OpenAIChat,
    build_llm,
    build_messages,
    parse_draft,
)
from ragkit.ground import normalise_for_match
from ragkit.models import Chunk, RetrievedChunk
from ragkit.prompts import PROMPT_VERSION, SYSTEM


def retrieved(*texts: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=Chunk(
                chunk_id=f"policy.md:{i:04d}", doc_id="policy.md", text=t, ordinal=i, page=i + 1
            ),
            score=0.9 - 0.1 * i,
        )
        for i, t in enumerate(texts)
    ]


class TestPrompt:
    def test_version_is_recorded(self):
        assert PROMPT_VERSION

    def test_passages_are_numbered_from_one(self):
        _, user = build_messages("How much notice?", retrieved("First text.", "Second text."))
        assert "[1] (source: policy.md p.1)" in user
        assert "[2] (source: policy.md p.2)" in user

    def test_question_is_included(self):
        _, user = build_messages("How much notice?", retrieved("text"))
        assert "Question: How much notice?" in user

    def test_system_prompt_demands_json_and_refusal(self):
        system, _ = build_messages("q", retrieved("text"))
        assert system == SYSTEM
        assert "sufficient" in system
        assert "verbatim" in system


class TestParseDraft:
    def test_plain_json(self):
        draft = parse_draft(
            json.dumps(
                {
                    "sufficient": True,
                    "answer": "Thirty days [1].",
                    "support": [{"passage": 1, "quote": "thirty days"}],
                }
            )
        )
        assert draft.sufficient is True
        assert draft.answer == "Thirty days [1]."
        assert draft.support == [{"passage": 1, "quote": "thirty days"}]

    def test_fenced_json(self):
        raw = '```json\n{"sufficient": true, "answer": "A [1].", "support": []}\n```'
        assert parse_draft(raw).answer == "A [1]."

    def test_unlabelled_fence(self):
        raw = '```\n{"sufficient": false, "answer": "", "support": []}\n```'
        assert parse_draft(raw).sufficient is False

    def test_json_wrapped_in_prose(self):
        raw = 'Sure! Here is the result:\n{"sufficient": true, "answer": "A [1].", "support": []}\nHope that helps.'
        assert parse_draft(raw).answer == "A [1]."

    def test_declining_response(self):
        draft = parse_draft('{"sufficient": false, "answer": "", "support": []}')
        assert draft.sufficient is False
        assert draft.support == []

    def test_non_dict_support_entries_are_dropped(self):
        raw = '{"sufficient": true, "answer": "A [1].", "support": [{"passage": 1, "quote": "x"}, "nonsense", 42]}'
        assert parse_draft(raw).support == [{"passage": 1, "quote": "x"}]

    def test_missing_support_defaults_to_empty(self):
        assert parse_draft('{"sufficient": true, "answer": "A."}').support == []

    def test_answer_is_stripped(self):
        assert parse_draft('{"sufficient": true, "answer": "  A [1].  "}').answer == "A [1]."

    @pytest.mark.parametrize(
        ("raw", "match"),
        [
            ("", "empty"),
            ("   ", "empty"),
            ("I cannot help with that.", "No JSON object"),
            ("{not json at all}", "not valid JSON"),
            ('["a", "list"]', "Expected a JSON object"),
            ('"just a string"', "Expected a JSON object"),
            ('{"answer": "A."}', "'sufficient' must be a boolean"),
            ('{"sufficient": "yes", "answer": "A."}', "'sufficient' must be a boolean"),
            ('{"sufficient": 1, "answer": "A."}', "'sufficient' must be a boolean"),
            ('{"sufficient": true, "answer": 42}', "'answer' must be a string"),
            ('{"sufficient": true, "answer": "A.", "support": "nope"}', "'support' must be a list"),
        ],
    )
    def test_rejections(self, raw, match):
        with pytest.raises(MalformedLLMResponseError, match=match):
            parse_draft(raw)

    def test_missing_sufficient_is_never_assumed_true(self):
        # Defaulting it would publish an answer the model may not stand behind.
        with pytest.raises(MalformedLLMResponseError):
            parse_draft('{"answer": "Thirty days [1].", "support": [{"passage": 1, "quote": "x"}]}')


class TestExtractiveStub:
    def test_identifies_itself_honestly(self):
        assert ExtractiveStub().model == "extractive-stub"

    def test_quote_is_verbatim_from_the_passage(self):
        # The stub's output must survive the same verification a real model's does; if it
        # paraphrased, every end-to-end test would pass for the wrong reason.
        passage = "Employees must give thirty days written notice before resigning."
        system, user = build_messages("How much notice before resigning?", retrieved(passage))
        draft = parse_draft(ExtractiveStub().complete(system=system, user=user))
        assert draft.sufficient is True
        quote = draft.support[0]["quote"]
        assert normalise_for_match(quote) in normalise_for_match(passage)

    def test_cites_the_passage_it_quoted(self):
        system, user = build_messages(
            "What happens to unused leave?",
            retrieved(
                "Employees must give thirty days notice.",
                "Unused annual leave expires on 31 March of the following year.",
            ),
        )
        draft = parse_draft(ExtractiveStub().complete(system=system, user=user))
        assert draft.support[0]["passage"] == 2
        assert "[2]" in draft.answer

    def test_declines_when_no_content_words_overlap(self):
        system, user = build_messages(
            "What is the capital of Mongolia?",
            retrieved("Employees must give thirty days written notice."),
        )
        assert parse_draft(ExtractiveStub().complete(system=system, user=user)).sufficient is False

    def test_declines_on_a_stopword_only_question(self):
        system, user = build_messages("what is the of and", retrieved("Notice period rules."))
        assert parse_draft(ExtractiveStub().complete(system=system, user=user)).sufficient is False

    def test_does_not_glue_a_markdown_heading_onto_the_sentence(self):
        passage = "# Notice Period\n\nEmployees must give thirty days written notice."
        system, user = build_messages("How much notice?", retrieved(passage))
        draft = parse_draft(ExtractiveStub().complete(system=system, user=user))
        assert "# Notice Period" not in draft.support[0]["quote"]

    def test_no_passages_declines(self):
        system, user = build_messages("anything", [])
        assert parse_draft(ExtractiveStub().complete(system=system, user=user)).sufficient is False


class TestFactory:
    @pytest.mark.parametrize("alias", ["echo", "stub", "extractive"])
    def test_builds_the_stub(self, alias):
        assert isinstance(build_llm(alias), ExtractiveStub)

    def test_unknown_name_is_an_error(self):
        with pytest.raises(GenerationError, match="Unknown LLM"):
            build_llm("llama-via-vibes")

    def test_openai_without_a_key_fails_clearly(self):
        pytest.importorskip("openai")
        with pytest.raises(GenerationError, match="OPENAI_API_KEY"):
            OpenAIChat(model="gpt-4o-mini", api_key=None)

    def test_anthropic_without_a_key_fails_clearly(self):
        pytest.importorskip("anthropic")
        with pytest.raises(GenerationError, match="ANTHROPIC_API_KEY"):
            AnthropicMessages(model="claude-sonnet-4-5", api_key=None)
