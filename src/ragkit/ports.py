"""The three things this package needs from the outside world.

Each is a `Protocol`, not a base class, so an implementation never imports ragkit to satisfy
it. That is what keeps `sentence-transformers`, `qdrant-client` and every provider SDK out of
the default install and out of the test suite: the fakes in `tests/conftest.py` are ordinary
classes that happen to have the right methods.

The protocols are also the honest boundary for what is *not* implemented here. Reranking,
for instance, would be a fourth protocol; there is no stub pretending otherwise.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragkit.models import Chunk, RetrievedChunk

# ---------------------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors.

    Documents and queries get separate methods because asymmetric models exist (E5, BGE and
    friends want `passage:` / `query:` prefixes). A single `encode` would make using one of
    those a silent accuracy loss rather than a type error.
    """

    @property
    def dimensions(self) -> int:
        """Vector length. Used to reject a store/embedder mismatch at startup."""
        ...

    @property
    def name(self) -> str:
        """Identifier recorded alongside the index, so a swap is traceable."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Stores chunks with their vectors, and searches them.

    `upsert` takes whole `Chunk` objects rather than vectors plus a side-table of metadata.
    The original version of this project stored vectors with no payload, which made the index
    structurally incapable of returning text — the bug that made its "grounded answers" claim
    false. Requiring the chunk here makes that mistake impossible to repeat.
    """

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """Write chunks. Returns the number written. Must be idempotent on `chunk_id`."""
        ...

    def search(self, vector: list[float], *, limit: int) -> list[RetrievedChunk]:
        """Nearest neighbours by cosine similarity, highest first."""
        ...

    def count(self) -> int: ...

    def clear(self) -> None: ...


@runtime_checkable
class LLM(Protocol):
    """Generates a completion from a prompt.

    Deliberately the narrowest useful surface: one string in, one string out. Everything that
    makes the answer trustworthy — the response contract, parsing, quote verification — is
    this package's job, not the provider's, so swapping providers cannot weaken it.
    """

    @property
    def model(self) -> str: ...

    def complete(self, *, system: str, user: str) -> str:
        """Return the model's raw text. Implementations should not retry; callers decide."""
        ...
