"""Generation: calling a model, and parsing what comes back.

Nothing here decides whether an answer is good — that is `ground.py`'s job. This module's
only responsibilities are to ask the question properly and to turn an untrusted string into a
`Draft` without crashing on the many shapes a model can return.

Three backends. `ExtractiveStub` is the default and is **not a language model**; it is a
deterministic baseline that lets the whole pipeline, its tests and the eval harness run
offline. OpenAI and Anthropic sit behind extras and satisfy the same `LLM` protocol.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ragkit.embeddings import content_tokens
from ragkit.errors import GenerationError, MalformedLLMResponseError
from ragkit.models import RetrievedChunk
from ragkit.prompts import SYSTEM, render_user


@dataclass(frozen=True)
class Draft:
    """A model's proposed answer, before verification."""

    sufficient: bool
    answer: str = ""
    support: list[dict] = field(default_factory=list)
    """Raw support entries, intentionally untyped.

    `ground.verify` is the component that decides what counts as evidence, so entries reach
    it exactly as the model produced them — including malformed ones, which it reports rather
    than silently dropping here.
    """


# ---------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_draft(raw: str) -> Draft:
    """Turn a model's reply into a `Draft`.

    Tolerates the two formatting habits that are near-universal and harmless — a ```json
    fence, and prose around the object — and rejects everything else loudly. Being lenient
    about *shape* here would mean guessing what the model meant, which is exactly the kind of
    silent repair that makes a pipeline untrustworthy.
    """
    if not raw or not raw.strip():
        raise MalformedLLMResponseError("The model returned an empty response")

    text = _FENCE.sub("", raw.strip())
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the outermost {...}, which recovers a model that prefixed its JSON
        # with a sentence of explanation.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise MalformedLLMResponseError(
                f"No JSON object in the response (began {text[:80]!r})"
            ) from None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise MalformedLLMResponseError(f"Response was not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise MalformedLLMResponseError(f"Expected a JSON object, got {type(payload).__name__}")

    sufficient = payload.get("sufficient")
    if not isinstance(sufficient, bool):
        raise MalformedLLMResponseError(
            f"'sufficient' must be a boolean, got {sufficient!r}. Treating a missing value as "
            f"true would publish an answer the model may not stand behind."
        )

    answer = payload.get("answer") or ""
    if not isinstance(answer, str):
        raise MalformedLLMResponseError(f"'answer' must be a string, got {type(answer).__name__}")

    support_raw = payload.get("support") or []
    if not isinstance(support_raw, list):
        raise MalformedLLMResponseError(
            f"'support' must be a list, got {type(support_raw).__name__}"
        )
    support = [e for e in support_raw if isinstance(e, dict)]

    return Draft(sufficient=sufficient, answer=answer.strip(), support=support)


def build_messages(question: str, retrieved: list[RetrievedChunk]) -> tuple[str, str]:
    """`(system, user)` for a question and its passages."""
    passages = [(i, r.chunk.label, r.chunk.text) for i, r in enumerate(retrieved, start=1)]
    return SYSTEM, render_user(question, passages)


# ---------------------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------------------

# Split on sentence punctuation *or* a blank line. Markdown headings carry no terminator, so
# sentence punctuation alone glued "# Leave and Notice Policy" onto the first real sentence
# and quoted the lot.
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n\s*\n+")
_PASSAGE_BLOCK = re.compile(
    r"^\[(\d+)\] \(source: .*?\)\n(.*?)(?=\n\n\[\d+\] \(source:|\Z)",
    re.MULTILINE | re.DOTALL,
)
_STUB_MIN_QUOTE = 12  # Mirrors ground.MIN_QUOTE_CHARS; a shorter quote cannot verify.


class ExtractiveStub:
    """A deterministic, offline baseline. **Not a language model.**

    It picks the passage with the most word overlap with the question, quotes the best
    sentence from it verbatim, and cites it. The answer is therefore extractive — copied, not
    written — so it reads stiffly and cannot synthesise across passages.

    It exists because a portfolio project that only runs with somebody's API key cannot be
    tested, evaluated or reviewed. Every assertion in the test suite and every number from
    `eval/run_eval.py` comes from this backend, which means they measure retrieval, chunking
    and verification rather than a vendor's model. Set `RAGKIT_LLM=openai` for real prose.
    """

    @property
    def model(self) -> str:
        return "extractive-stub"

    def complete(self, *, system: str, user: str) -> str:
        question, passages = self._parse_prompt(user)
        # Content words only, for the same reason the embedder filters them: shared function
        # words made an unanswerable question look supported.
        q_words = set(content_tokens(question))
        if not q_words or not passages:
            return json.dumps({"sufficient": False, "answer": "", "support": []})

        best: tuple[int, int, str] | None = None  # (overlap, number, quote)
        for number, text in passages:
            quote = self._best_span(text, q_words)
            if quote is None:
                continue
            overlap = len(q_words & set(content_tokens(quote)))
            if overlap and (best is None or overlap > best[0]):
                best = (overlap, number, quote)

        if best is None:
            return json.dumps({"sufficient": False, "answer": "", "support": []})

        _, number, quote = best
        return json.dumps(
            {
                "sufficient": True,
                "answer": f"{quote} [{number}]",
                "support": [{"passage": number, "quote": quote}],
            }
        )

    @staticmethod
    def _parse_prompt(user: str) -> tuple[str, list[tuple[int, str]]]:
        question_match = re.search(r"^Question: (.*)$", user, re.MULTILINE)
        question = question_match.group(1).strip() if question_match else ""
        passages = [(int(n), body.strip()) for n, body in _PASSAGE_BLOCK.findall(user)]
        return question, passages

    @staticmethod
    def _best_span(text: str, q_words: set[str]) -> str | None:
        """The sentence sharing most words with the question, lengthened if too short."""
        sentences = [s.strip() for s in _SENTENCE.split(text) if s.strip()]
        if not sentences:
            return None
        # Rank by question-word overlap, then by length, then by position. The length
        # tie-break matters: a Markdown heading like "# Notice Period" matches `notice` just
        # as well as the sentence that actually answers the question, and preferring the
        # earlier match would quote the heading. Among equally-matching spans the longer one
        # carries the information. Position last, so the result is deterministic.
        scored = sorted(
            (
                (len(q_words & set(content_tokens(s))), len(s), -i, s)
                for i, s in enumerate(sentences)
            ),
            reverse=True,
        )
        overlap, _length, neg_index, sentence = scored[0]
        if not overlap:
            return None
        if len(sentence) >= _STUB_MIN_QUOTE:
            return sentence
        # Too short to be acceptable evidence; extend with the following sentence(s) so the
        # quote is substantial, keeping it verbatim and contiguous.
        index = -neg_index
        extended = " ".join(sentences[index : index + 3])
        return extended if len(extended) >= _STUB_MIN_QUOTE else None


class OpenAIChat:
    """OpenAI chat completions. Needs the `openai` extra.

    Uses JSON mode, so the response contract is enforced by the provider as well as by
    `parse_draft`. Belt and braces: JSON mode guarantees syntax, never semantics.
    """

    def __init__(self, *, model: str, api_key: str | None, max_tokens: int = 700) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise GenerationError(
                "openai is not installed. `pip install '.[openai]'`, or set RAGKIT_LLM=echo."
            ) from exc
        if not api_key:
            raise GenerationError("RAGKIT_OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._model

    def complete(self, *, system: str, user: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                # Zero temperature: the task is extraction and attribution, where creativity
                # is purely a source of unverifiable claims.
                temperature=0.0,
                max_tokens=self._max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise GenerationError(f"OpenAI request failed: {exc}") from exc
        return response.choices[0].message.content or ""


class AnthropicMessages:
    """Anthropic messages API. Needs the `anthropic` extra."""

    def __init__(self, *, model: str, api_key: str | None, max_tokens: int = 700) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise GenerationError(
                "anthropic is not installed. `pip install '.[anthropic]'`."
            ) from exc
        if not api_key:
            raise GenerationError("RAGKIT_ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._model

    def complete(self, *, system: str, user: str) -> str:
        try:
            message = self._client.messages.create(
                model=self._model,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:
            raise GenerationError(f"Anthropic request failed: {exc}") from exc
        return "".join(block.text for block in message.content if block.type == "text")


def build_llm(
    kind: str,
    *,
    model: str = "",
    openai_api_key: str | None = None,
    anthropic_api_key: str | None = None,
    max_tokens: int = 700,
) -> object:
    """Resolve the configured LLM name to an instance."""
    if kind in {"echo", "stub", "extractive"}:
        return ExtractiveStub()
    if kind == "openai":
        return OpenAIChat(
            model=model or "gpt-4o-mini", api_key=openai_api_key, max_tokens=max_tokens
        )
    if kind == "anthropic":
        return AnthropicMessages(
            model=model or "claude-sonnet-4-5", api_key=anthropic_api_key, max_tokens=max_tokens
        )
    raise GenerationError(f"Unknown LLM '{kind}'. Use 'echo', 'openai' or 'anthropic'.")
