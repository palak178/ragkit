"""The data that moves between stages.

Every boundary in this package is a Pydantic model rather than a dict, for one practical
reason: the failure this project exists to prevent is an answer that quietly loses its link
back to the source text. A typed `Citation` that cannot be constructed without a `chunk_id`
makes that link structural instead of conventional.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------------------
# Documents and chunks
# ---------------------------------------------------------------------------------------


class Page(BaseModel):
    """One page of extracted text, before chunking."""

    model_config = ConfigDict(frozen=True)

    number: int = Field(ge=1, description="1-based page number, as a reader would cite it")
    text: str


class Chunk(BaseModel):
    """A retrievable unit of text, with enough provenance to cite it.

    `page` survives chunking on purpose. A citation that can only say "somewhere in this
    document" is not auditable, and page number is the coordinate humans actually use.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    text: str
    ordinal: int = Field(ge=0, description="Position of this chunk within its document")
    page: int | None = Field(default=None, description="Source page, when the format has pages")

    @property
    def label(self) -> str:
        """Human-readable source, e.g. `handbook.pdf p.4`."""
        return f"{self.doc_id} p.{self.page}" if self.page is not None else self.doc_id


class RetrievedChunk(BaseModel):
    """A chunk plus the similarity that retrieved it."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float = Field(description="Cosine similarity in [-1, 1]; 1.0 is identical")


# ---------------------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------------------


class Citation(BaseModel):
    """A validated link from a sentence in the answer to the text that supports it.

    A `Citation` only ever exists for a quote that deterministic code has already confirmed
    appears in the cited chunk. Unverified support never reaches this type.
    """

    model_config = ConfigDict(frozen=True)

    marker: int = Field(ge=1, description="The [n] shown in the answer text")
    chunk_id: str
    doc_id: str
    page: int | None
    quote: str = Field(description="The verbatim span from the chunk that supports the claim")


Verdict = Literal["grounded", "ungrounded", "insufficient_retrieval", "model_declined"]


class GroundednessReport(BaseModel):
    """Why an answer was published or refused, in numbers rather than prose.

    This is the audit record. Every field is produced by deterministic code, never by the
    model, so it can be trusted to describe the model rather than merely repeat it.
    """

    model_config = ConfigDict(frozen=True)

    verdict: Verdict
    best_score: float = Field(description="Top retrieval score; -1.0 when nothing was retrieved")
    score_threshold: float
    chunks_retrieved: int
    support_claimed: int = Field(default=0, description="Supporting quotes the model offered")
    support_verified: int = Field(default=0, description="…of which were found in the cited chunk")
    rejected_out_of_range: list[int] = Field(
        default_factory=list, description="Passage indices the model cited that were never supplied"
    )
    rejected_quote_not_found: list[int] = Field(
        default_factory=list, description="Passage indices whose quote was not in the chunk"
    )

    @property
    def is_grounded(self) -> bool:
        return self.verdict == "grounded"


class Answer(BaseModel):
    """The result of asking a question.

    `text` is `None` whenever `groundedness.verdict != "grounded"`. There is deliberately no
    third state where prose is returned alongside a failed check — a refusal that still shows
    the ungrounded draft is just a hallucination with a disclaimer.
    """

    question: str
    text: str | None
    citations: list[Citation] = Field(default_factory=list)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    groundedness: GroundednessReport
    refusal_reason: str | None = None

    @property
    def refused(self) -> bool:
        return self.text is None

    def rendered(self) -> str:
        """The answer with a numbered source list, as a CLI or chat UI would show it."""
        if self.text is None:
            return f"No grounded answer. {self.refusal_reason or ''}".strip()
        lines = [self.text, ""]
        for c in self.citations:
            lines.append(f"[{c.marker}] {c.doc_id}" + (f" p.{c.page}" if c.page else ""))
        return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------------------


class IngestReport(BaseModel):
    """What one ingest call did, including the parts that were skipped."""

    doc_id: str
    pages: int
    chunks_written: int
    chars_extracted: int
    skipped_empty_pages: list[int] = Field(default_factory=list)
