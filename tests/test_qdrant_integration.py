"""Qdrant tests against a real server. Skipped unless one is reachable.

There is deliberately no mocked Qdrant anywhere in this suite. A mock would assert that
`upsert` was called with the arguments the test already chose, which proves nothing about the
two things that actually go wrong with a vector database: whether the payload comes back, and
whether a re-upsert replaces a point or duplicates it. Both need a real server, so these tests
skip loudly instead of passing cheaply.

    docker run -p 6333:6333 qdrant/qdrant
    RAGKIT_QDRANT_URL=http://localhost:6333 pytest tests/test_qdrant_integration.py
"""

from __future__ import annotations

import os
import uuid

import pytest

from ragkit.models import Chunk

QDRANT_URL = os.getenv("RAGKIT_QDRANT_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not QDRANT_URL,
        reason="RAGKIT_QDRANT_URL is not set; start Qdrant and set it to run these",
    ),
]


@pytest.fixture
def qdrant_store():
    """A store backed by a throwaway collection, deleted afterwards."""
    pytest.importorskip("qdrant_client")
    from ragkit.store import QdrantStore

    collection = f"ragkit_test_{uuid.uuid4().hex[:12]}"
    store = QdrantStore(
        url=QDRANT_URL or "",
        collection=collection,
        dimensions=4,
        api_key=os.getenv("RAGKIT_QDRANT_API_KEY"),
    )
    try:
        yield store
    finally:
        store._client.delete_collection(collection)


def chunk(cid: str, text: str, page: int | None = 3) -> Chunk:
    return Chunk(chunk_id=cid, doc_id="policy.md", text=text, ordinal=0, page=page)


def test_payload_round_trips_the_text(qdrant_store):
    """The regression test for the bug that made this project's original claim false."""
    qdrant_store.upsert(
        [chunk("policy.md:0000", "Employees must give thirty days written notice.")],
        [[1.0, 0.0, 0.0, 0.0]],
    )
    hit = qdrant_store.search([1.0, 0.0, 0.0, 0.0], limit=1)[0]
    assert hit.chunk.text == "Employees must give thirty days written notice."
    assert hit.chunk.chunk_id == "policy.md:0000"
    assert hit.chunk.doc_id == "policy.md"
    assert hit.chunk.page == 3


def test_reupsert_replaces_rather_than_duplicating(qdrant_store):
    """The other original bug, inverted: every point was written with a hard-coded id."""
    qdrant_store.upsert([chunk("policy.md:0000", "first")], [[1.0, 0.0, 0.0, 0.0]])
    qdrant_store.upsert([chunk("policy.md:0000", "second")], [[1.0, 0.0, 0.0, 0.0]])
    assert qdrant_store.count() == 1
    assert qdrant_store.search([1.0, 0.0, 0.0, 0.0], limit=1)[0].chunk.text == "second"


def test_distinct_chunks_coexist(qdrant_store):
    qdrant_store.upsert(
        [chunk("a:0000", "alpha"), chunk("b:0000", "beta")],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )
    assert qdrant_store.count() == 2


def test_nullable_page_survives_the_round_trip(qdrant_store):
    # Plain-text documents have no page. `None` must come back as `None`, not 0.
    qdrant_store.upsert(
        [chunk("notes.md:0000", "a note with no pages", page=None)], [[0.5, 0.5, 0.5, 0.5]]
    )
    assert qdrant_store.search([0.5, 0.5, 0.5, 0.5], limit=1)[0].chunk.page is None


def test_search_orders_by_similarity(qdrant_store):
    qdrant_store.upsert(
        [chunk("d:0000", "near"), chunk("d:0001", "far")],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    )
    results = qdrant_store.search([1.0, 0.0, 0.0, 0.0], limit=2)
    assert [r.chunk.text for r in results] == ["near", "far"]
    assert results[0].score > results[1].score


def test_wrong_dimensions_are_rejected_on_an_existing_collection(qdrant_store):
    """Changing embedder without re-indexing must fail loudly, not rank nonsense."""
    from ragkit.errors import DimensionMismatchError
    from ragkit.store import QdrantStore

    with pytest.raises(DimensionMismatchError, match="Re-index"):
        QdrantStore(
            url=QDRANT_URL or "",
            collection=qdrant_store._collection,
            dimensions=768,
            api_key=os.getenv("RAGKIT_QDRANT_API_KEY"),
        )


def test_clear_empties_the_collection(qdrant_store):
    qdrant_store.upsert([chunk("d:0000", "text")], [[1.0, 0.0, 0.0, 0.0]])
    qdrant_store.clear()
    assert qdrant_store.count() == 0
