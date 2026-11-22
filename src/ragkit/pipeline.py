"""Orchestration: the two things a caller actually wants to do.

`ingest` puts a document in the index. `ask` answers a question or explains why it will not.
Both are thin, because the decisions live in `chunking`, `ground` and the stores — this module
only sequences them and owns the one piece of control flow worth arguing about: what happens
when the model returns something unparseable.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ragkit import ground
from ragkit.chunking import chunk_document
from ragkit.config import Settings, load_settings
from ragkit.embeddings import build_embedder
from ragkit.errors import MalformedLLMResponseError
from ragkit.extract import extract
from ragkit.generate import build_llm, build_messages, parse_draft
from ragkit.models import Answer, GroundednessReport, IngestReport
from ragkit.ports import LLM, Embedder, VectorStore
from ragkit.store import InMemoryStore, QdrantStore

log = logging.getLogger("ragkit")

MALFORMED_RETRIES = 1
"""One retry on an unparseable response, then give up.

A malformed reply is usually a sampling accident and often parses on a second attempt, so one
retry is worth the latency. More than one is throwing money at a prompt that is not working,
and it hides the problem from whoever should be fixing it.
"""


class RagPipeline:
    """Ingest documents; answer questions from them, with citations or not at all."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VectorStore,
        llm: LLM,
        settings: Settings | None = None,
    ) -> None:
        self.settings = (settings or load_settings()).validated()
        self.embedder = embedder
        self.store = store
        self.llm = llm

    # -- construction -------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> RagPipeline:
        """Build a pipeline from configuration, resolving every backend by name."""
        cfg = (settings or load_settings()).validated()
        embedder = build_embedder(
            cfg.embedder,
            dimensions=cfg.embedding_dimensions,
            model_name=cfg.embedding_model,
        )
        store: VectorStore
        if cfg.store == "qdrant":
            store = QdrantStore(
                url=cfg.qdrant_url or "",
                collection=cfg.qdrant_collection,
                dimensions=embedder.dimensions,  # type: ignore[attr-defined]
                api_key=cfg.qdrant_api_key,
            )
        else:
            store = InMemoryStore()
        llm = build_llm(
            cfg.llm,
            model=cfg.llm_model,
            openai_api_key=cfg.openai_api_key,
            anthropic_api_key=cfg.anthropic_api_key,
            max_tokens=cfg.max_answer_tokens,
        )
        return cls(embedder=embedder, store=store, llm=llm, settings=cfg)  # type: ignore[arg-type]

    # -- ingest -------------------------------------------------------------------------

    def ingest(self, path: str | Path, *, doc_id: str | None = None) -> IngestReport:
        """Extract, chunk, embed and store one document.

        Re-ingesting the same document replaces its chunks rather than duplicating them,
        because `chunk_id` is derived from the document id and the chunk's ordinal and both
        stores upsert on it.
        """
        doc = extract(path, doc_id=doc_id)
        chunks = chunk_document(
            doc,
            chunk_chars=self.settings.chunk_chars,
            overlap_chars=self.settings.chunk_overlap_chars,
        )
        written = 0
        if chunks:
            vectors = self.embedder.embed_documents([c.text for c in chunks])
            written = self.store.upsert(chunks, vectors)

        report = IngestReport(
            doc_id=doc.doc_id,
            pages=len(doc.pages),
            chunks_written=written,
            chars_extracted=doc.chars,
            skipped_empty_pages=[p.number for p in doc.pages if not p.text.strip()],
        )
        log.info(
            "ingested %s: %d pages, %d chunks, %d chars",
            report.doc_id,
            report.pages,
            report.chunks_written,
            report.chars_extracted,
        )
        return report

    # -- ask ----------------------------------------------------------------------------

    def ask(self, question: str) -> Answer:
        """Answer `question` from the index, or refuse and say why."""
        question = question.strip()
        if not question:
            return self._refuse(
                question,
                GroundednessReport(
                    verdict="insufficient_retrieval",
                    best_score=-1.0,
                    score_threshold=self.settings.score_threshold,
                    chunks_retrieved=0,
                ),
                retrieved=[],
            )

        query_vector = self.embedder.embed_query(question)
        retrieved = self.store.search(query_vector, limit=self.settings.top_k)

        # Gate 1 — deterministic, before any token is spent.
        gated = ground.gate_retrieval(retrieved, threshold=self.settings.score_threshold)
        if gated is not None:
            log.info("refused %r: %s (best=%.3f)", question, gated.verdict, gated.best_score)
            return self._refuse(question, gated, retrieved)

        system, user = build_messages(question, retrieved)
        draft = self._draft(system=system, user=user)

        if draft is None or not draft.sufficient:
            report = GroundednessReport(
                verdict="model_declined",
                best_score=retrieved[0].score,
                score_threshold=self.settings.score_threshold,
                chunks_retrieved=len(retrieved),
            )
            return self._refuse(question, report, retrieved)

        # Gate 2 — deterministic, on what the model actually produced.
        report, citations = ground.verify(
            answer=draft.answer,
            support=draft.support,
            retrieved=retrieved,
            threshold=self.settings.score_threshold,
        )
        if not report.is_grounded:
            log.info(
                "withheld answer to %r: %d/%d quotes verified",
                question,
                report.support_verified,
                report.support_claimed,
            )
            return self._refuse(question, report, retrieved)

        return Answer(
            question=question,
            text=draft.answer,
            citations=citations,
            retrieved=retrieved,
            groundedness=report,
        )

    def _draft(self, *, system: str, user: str):
        """Call the model, retrying once if the response will not parse."""
        last: MalformedLLMResponseError | None = None
        for attempt in range(MALFORMED_RETRIES + 1):
            raw = self.llm.complete(system=system, user=user)
            try:
                return parse_draft(raw)
            except MalformedLLMResponseError as exc:
                last = exc
                log.warning("unparseable model response (attempt %d): %s", attempt + 1, exc)
        # Treated as a refusal rather than an exception: from a caller's point of view a
        # model that will not produce the contract is indistinguishable from one that cannot
        # answer, and both should end in "no grounded answer" rather than a 500.
        log.error("giving up after %d malformed responses: %s", MALFORMED_RETRIES + 1, last)
        return None

    def _refuse(self, question: str, report: GroundednessReport, retrieved: list) -> Answer:
        return Answer(
            question=question,
            text=None,
            citations=[],
            retrieved=retrieved,
            groundedness=report,
            refusal_reason=ground.refusal_text(report),
        )
