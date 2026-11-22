"""Vector stores.

Both implementations store the chunk **text** alongside its vector. That sounds too obvious
to mention, except that the version of this project being replaced wrote
`PointStruct(id=1, vector=...)` with no payload at all: the index physically could not return
text, so "retrieval-grounded answers" was impossible by construction, and the hard-coded `id=1`
meant every ingested document overwrote the previous one. Both are fixed here, and the
`VectorStore` protocol is shaped so neither can recur — it takes `Chunk` objects, not vectors.
"""

from __future__ import annotations

import math
import uuid

from ragkit.errors import DimensionMismatchError, StoreError
from ragkit.models import Chunk, RetrievedChunk


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity, without assuming either vector is normalised.

    The embedders here do normalise, which would make a dot product sufficient — but a store
    that silently depends on that breaks the moment someone plugs in a backend that does not,
    and it breaks as quietly degraded ranking rather than as an error.
    """
    if len(a) != len(b):
        raise DimensionMismatchError(f"Vector lengths differ: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0  # An empty query matches nothing, rather than raising.
    return dot / (norm_a * norm_b)


class InMemoryStore:
    """Exact brute-force search over a dict. The default, and the one tests use.

    Brute force is the right choice at this scale and an honest one: for a few thousand
    chunks an exhaustive scan is milliseconds and returns *exact* neighbours, so retrieval
    quality measured against it is the pipeline's quality rather than an index's recall
    curve. Approximate indexes earn their complexity above roughly a million vectors, which
    is what `QdrantStore` is for.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}
        self._dimensions: int | None = None

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise StoreError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        for chunk, vector in zip(chunks, vectors, strict=True):
            if self._dimensions is None:
                self._dimensions = len(vector)
            elif len(vector) != self._dimensions:
                raise DimensionMismatchError(
                    f"Collection holds {self._dimensions}-d vectors but got {len(vector)}. "
                    f"This usually means the embedder changed without re-indexing."
                )
            # Keyed on chunk_id, so re-ingesting a document replaces its chunks instead of
            # duplicating them.
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = list(vector)
        return len(chunks)

    def search(self, vector: list[float], *, limit: int) -> list[RetrievedChunk]:
        if limit < 1:
            raise StoreError(f"limit must be >= 1, got {limit}")
        scored = [
            RetrievedChunk(chunk=self._chunks[cid], score=cosine_similarity(vector, vec))
            for cid, vec in self._vectors.items()
        ]
        # Tie-break on chunk_id so equal scores order deterministically; otherwise test
        # expectations and eval numbers wobble between runs for no real reason.
        scored.sort(key=lambda r: (-r.score, r.chunk.chunk_id))
        return scored[:limit]

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._vectors.clear()
        self._dimensions = None


class QdrantStore:
    """Qdrant-backed store. Needs the `qdrant` extra.

    Point IDs are UUIDv5 derived from `chunk_id`, which gives Qdrant the UUID it wants while
    keeping the mapping deterministic — so an upsert of the same chunk overwrites rather than
    duplicating, and a re-ingest is idempotent.
    """

    _NAMESPACE = uuid.UUID("6b6cf1aa-6a17-5b1f-9a2e-6a8b1d4e2f10")

    def __init__(
        self,
        *,
        url: str,
        collection: str,
        dimensions: int,
        api_key: str | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise StoreError(
                "qdrant-client is not installed. `pip install '.[qdrant]'`, or set "
                "RAGKIT_STORE=memory."
            ) from exc
        self._client = QdrantClient(url=url, api_key=api_key)
        self._collection = collection
        self._dimensions = dimensions
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dimensions, distance=Distance.COSINE),
            )
            return
        # An existing collection with the wrong width would accept writes and return
        # nonsense, so check rather than assume.
        info = self._client.get_collection(self._collection)
        existing = info.config.params.vectors.size  # type: ignore[union-attr]
        if existing != self._dimensions:
            raise DimensionMismatchError(
                f"Collection '{self._collection}' expects {existing}-d vectors but the "
                f"configured embedder produces {self._dimensions}-d. Re-index into a new "
                f"collection, or change RAGKIT_EMBEDDER back."
            )

    def _point_id(self, chunk_id: str) -> str:
        return str(uuid.uuid5(self._NAMESPACE, chunk_id))

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        from qdrant_client.models import PointStruct

        if len(chunks) != len(vectors):
            raise StoreError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        points = [
            PointStruct(
                id=self._point_id(chunk.chunk_id),
                vector=list(vector),
                # The payload carries the text. This is the line whose absence made the
                # original project's central claim false.
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "ordinal": chunk.ordinal,
                    "page": chunk.page,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, points=points, wait=True)
        return len(points)

    def search(self, vector: list[float], *, limit: int) -> list[RetrievedChunk]:
        hits = self._client.query_points(
            collection_name=self._collection,
            query=list(vector),
            limit=limit,
            with_payload=True,
        ).points
        results: list[RetrievedChunk] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                RetrievedChunk(
                    chunk=Chunk(
                        chunk_id=payload["chunk_id"],
                        doc_id=payload["doc_id"],
                        text=payload["text"],
                        ordinal=payload["ordinal"],
                        page=payload.get("page"),
                    ),
                    score=float(hit.score),
                )
            )
        return results

    def count(self) -> int:
        return int(self._client.count(self._collection, exact=True).count)

    def clear(self) -> None:
        self._client.delete_collection(self._collection)
        self._ensure_collection()
