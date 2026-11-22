"""Shared fixtures and fakes.

Everything here is offline and deterministic. No fixture downloads a model, opens a socket or
reads an API key, which is what makes the suite usable as a pre-commit check rather than a
nightly job — and what makes its assertions about *this* code rather than about a vendor's
model on the day it ran.

The fakes are plain classes. They satisfy `ragkit.ports` by having the right methods, not by
inheriting anything, which is the point of using Protocols there.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragkit.config import Settings
from ragkit.embeddings import HashingEmbedder
from ragkit.generate import ExtractiveStub
from ragkit.pipeline import RagPipeline
from ragkit.store import InMemoryStore

# ---------------------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------------------

# Deliberately written so that distinct questions have distinct best chunks, and so that one
# fact (the seven-day probation notice) contradicts the headline rule — a corpus where every
# chunk says roughly the same thing cannot detect a retrieval regression.
NOTICE_POLICY = """\
# Leave and Notice Policy

## Notice period

Employees must give thirty days written notice before resigning from the company. The notice
period begins on the date the written resignation is received by the human resources
department, not the date it was signed.

## Annual leave

Full-time employees accrue twenty-one days of paid annual leave in each calendar year.
Unused annual leave expires on 31 March of the following year and is not paid out.

## Probation

The probation period lasts six months from the start date. During probation the required
notice is seven days rather than thirty days.

## Equipment

Company laptops must be returned on the final working day. A replacement fee of 400 pounds
applies to equipment that is not returned.
"""


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """A directory holding the sample documents."""
    (tmp_path / "notice_policy.md").write_text(NOTICE_POLICY, encoding="utf-8")
    (tmp_path / "empty.txt").write_text("   \n\n  ", encoding="utf-8")
    return tmp_path


@pytest.fixture
def policy_doc(corpus_dir: Path) -> Path:
    return corpus_dir / "notice_policy.md"


# ---------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------


@pytest.fixture
def settings() -> Settings:
    """Test settings: small chunks so one short document still exercises retrieval.

    The threshold is lower than the shipped default because the hashing embedder is lexical
    and its similarities are smaller than a transformer's. Tests that care about the gate set
    their own value explicitly rather than depending on this one.
    """
    return Settings(
        chunk_chars=320,
        chunk_overlap_chars=60,
        top_k=3,
        score_threshold=0.05,
        embedder="hashing",
        embedding_dimensions=512,
        store="memory",
        llm="echo",
    )


# ---------------------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------------------


class ScriptedLLM:
    """Returns pre-written raw strings, one per call.

    This is how the verification paths get tested: a real model cannot be made to fabricate a
    quote on demand, but a script can, so `ground.verify`'s rejection branches are reachable
    from the pipeline rather than only in isolation.
    """

    def __init__(self, *responses: str) -> None:
        if not responses:
            raise ValueError("ScriptedLLM needs at least one response")
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    @property
    def model(self) -> str:
        return "scripted"

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        # The last response repeats, so a retry path can be given one bad reply and then
        # exercise "still bad" without listing it twice.
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[index]


def draft_json(
    *, sufficient: bool = True, answer: str = "", support: list[dict] | None = None
) -> str:
    """Build a well-formed response body, for use with `ScriptedLLM`."""
    return json.dumps({"sufficient": sufficient, "answer": answer, "support": support or []})


class FixedEmbedder:
    """Returns a constant vector of a chosen width. For dimension-mismatch tests."""

    def __init__(self, dimensions: int, fill: float = 1.0) -> None:
        self._dimensions = dimensions
        self._fill = fill

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def name(self) -> str:
        return f"fixed-{self._dimensions}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[self._fill] * self._dimensions for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [self._fill] * self._dimensions


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dimensions=512)


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def pipeline(embedder, store, settings) -> RagPipeline:
    """A complete, fully offline pipeline using the extractive stub."""
    return RagPipeline(embedder=embedder, store=store, llm=ExtractiveStub(), settings=settings)


@pytest.fixture
def indexed_pipeline(pipeline: RagPipeline, policy_doc: Path) -> RagPipeline:
    pipeline.ingest(policy_doc)
    return pipeline


def make_pipeline(embedder, store, settings, llm) -> RagPipeline:
    """Assemble a pipeline with a specific LLM. Used where the fixture's stub is wrong."""
    return RagPipeline(embedder=embedder, store=store, llm=llm, settings=settings)


# ---------------------------------------------------------------------------------------
# A real PDF, built by hand
# ---------------------------------------------------------------------------------------


def minimal_pdf(text: str) -> bytes:
    """A single-page PDF containing `text`, with a correct xref table.

    Written out byte by byte rather than pulled from a binary fixture file, for two reasons:
    a reviewer can see exactly what is being tested, and the PDF extraction path gets
    exercised against a genuine PDF instead of a mock that would pass regardless of whether
    `pypdf` works at all.
    """
    escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    content = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET\n".encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length "
        + str(len(content)).encode("latin-1")
        + b" >>\nstream\n"
        + content
        + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("latin-1") + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n".encode(
        "latin-1"
    )
    out += b"%%EOF\n"
    return bytes(out)


@pytest.fixture
def pdf_path(tmp_path: Path) -> Path:
    path = tmp_path / "sample.pdf"
    path.write_bytes(minimal_pdf("Employees must give thirty days written notice."))
    return path
