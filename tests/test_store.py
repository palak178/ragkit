"""Tests for the in-memory store.

Two of these exist specifically because of the bugs in the project this replaces: the index
must return chunk **text** (it previously stored vectors with no payload) and re-ingesting a
document must replace its chunks rather than overwrite an unrelated one (every point was
written with a hard-coded `id=1`).

`QdrantStore` is exercised by `tests/test_qdrant_integration.py`, which skips without a
reachable server; there is no mock of it here, because a mocked vector database asserts only
that the mock was called.
"""

from __future__ import annotations

import pytest

from ragkit.errors import DimensionMismatchError, StoreError
from ragkit.models import Chunk
from ragkit.store import InMemoryStore, cosine_similarity


def chunk(cid: str, text: str = "some text", page: int | None = 1) -> Chunk:
    return Chunk(chunk_id=cid, doc_id=cid.split(":")[0], text=text, ordinal=0, page=page)


class TestCosine:
    def test_identical_vectors(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_magnitude_does_not_matter(self):
        # The store must not depend on its inputs being pre-normalised.
        assert cosine_similarity([3.0, 4.0], [6.0, 8.0]) == pytest.approx(1.0)

    def test_zero_vector_scores_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_length_mismatch_is_an_error(self):
        with pytest.raises(DimensionMismatchError):
            cosine_similarity([1.0], [1.0, 2.0])


class TestUpsert:
    def test_writes_and_counts(self, store: InMemoryStore):
        assert store.upsert([chunk("d:0", "a"), chunk("d:1", "b")], [[1.0, 0.0], [0.0, 1.0]]) == 2
        assert store.count() == 2

    def test_search_returns_the_chunk_text(self, store: InMemoryStore):
        # The bug this project was rewritten to fix: an index that cannot return text makes
        # grounded answers impossible by construction.
        store.upsert([chunk("d:0", "Employees must give thirty days notice.")], [[1.0, 0.0]])
        hit = store.search([1.0, 0.0], limit=1)[0]
        assert hit.chunk.text == "Employees must give thirty days notice."
        assert hit.chunk.page == 1
        assert hit.score == pytest.approx(1.0)

    def test_reupsert_replaces_rather_than_duplicates(self, store: InMemoryStore):
        store.upsert([chunk("d:0", "first version")], [[1.0, 0.0]])
        store.upsert([chunk("d:0", "second version")], [[1.0, 0.0]])
        assert store.count() == 1
        assert store.search([1.0, 0.0], limit=1)[0].chunk.text == "second version"

    def test_distinct_documents_coexist(self):
        # The hard-coded `id=1` bug meant each upload destroyed the last.
        store = InMemoryStore()
        store.upsert([chunk("a:0", "alpha")], [[1.0, 0.0]])
        store.upsert([chunk("b:0", "beta")], [[0.0, 1.0]])
        assert store.count() == 2
        assert {r.chunk.text for r in store.search([1.0, 1.0], limit=2)} == {"alpha", "beta"}

    def test_mismatched_lengths_are_rejected(self, store: InMemoryStore):
        with pytest.raises(StoreError, match="2 chunks but 1 vectors"):
            store.upsert([chunk("d:0"), chunk("d:1")], [[1.0]])

    def test_changing_dimensions_is_rejected(self, store: InMemoryStore):
        store.upsert([chunk("d:0")], [[1.0, 0.0]])
        with pytest.raises(DimensionMismatchError, match="re-index"):
            store.upsert([chunk("d:1")], [[1.0, 0.0, 0.0]])

    def test_empty_upsert_is_a_no_op(self, store: InMemoryStore):
        assert store.upsert([], []) == 0
        assert store.count() == 0


class TestSearch:
    def test_orders_by_descending_score(self, store: InMemoryStore):
        store.upsert(
            [chunk("d:0", "near"), chunk("d:1", "far"), chunk("d:2", "middle")],
            [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
        )
        results = store.search([1.0, 0.0], limit=3)
        assert [r.chunk.text for r in results] == ["near", "middle", "far"]
        assert results[0].score >= results[1].score >= results[2].score

    def test_limit_truncates(self, store: InMemoryStore):
        store.upsert([chunk(f"d:{i}") for i in range(5)], [[1.0, 0.0]] * 5)
        assert len(store.search([1.0, 0.0], limit=2)) == 2

    def test_limit_larger_than_the_index(self, store: InMemoryStore):
        store.upsert([chunk("d:0")], [[1.0, 0.0]])
        assert len(store.search([1.0, 0.0], limit=50)) == 1

    def test_empty_index_returns_nothing(self, store: InMemoryStore):
        assert store.search([1.0, 0.0], limit=5) == []

    def test_zero_limit_is_rejected(self, store: InMemoryStore):
        with pytest.raises(StoreError, match=">= 1"):
            store.search([1.0, 0.0], limit=0)

    def test_ties_break_deterministically(self, store: InMemoryStore):
        # Identical vectors score identically; without a tie-break, ordering would vary and
        # eval numbers would wobble between runs for no real reason.
        store.upsert([chunk(f"d:{i}") for i in range(4)], [[1.0, 0.0]] * 4)
        first = [r.chunk.chunk_id for r in store.search([1.0, 0.0], limit=4)]
        assert first == sorted(first)
        assert first == [r.chunk.chunk_id for r in store.search([1.0, 0.0], limit=4)]

    def test_search_rejects_wrong_width(self, store: InMemoryStore):
        store.upsert([chunk("d:0")], [[1.0, 0.0]])
        with pytest.raises(DimensionMismatchError):
            store.search([1.0, 0.0, 0.0], limit=1)


class TestClear:
    def test_clear_empties_and_resets_dimensions(self, store: InMemoryStore):
        store.upsert([chunk("d:0")], [[1.0, 0.0]])
        store.clear()
        assert store.count() == 0
        assert store.search([1.0, 0.0], limit=1) == []
        # Width is forgotten too, so a different embedder can be used after a clear.
        store.upsert([chunk("d:1")], [[1.0, 0.0, 0.0]])
        assert store.count() == 1
