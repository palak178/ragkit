"""ragkit — grounded retrieval-augmented question answering.

The contract of this package in one sentence: an answer comes back with citations whose
quotes have been mechanically verified against the retrieved text, or it does not come back
at all.

    from ragkit.pipeline import RagPipeline

    pipeline = RagPipeline.from_settings()
    pipeline.ingest("handbook.pdf")
    answer = pipeline.ask("How much notice is required?")

    if answer.refused:
        print(answer.refusal_reason)
    else:
        print(answer.rendered())
"""

from __future__ import annotations

from ragkit.models import (
    Answer,
    Chunk,
    Citation,
    GroundednessReport,
    IngestReport,
    Page,
    RetrievedChunk,
)
from ragkit.pipeline import RagPipeline

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "Chunk",
    "Citation",
    "GroundednessReport",
    "IngestReport",
    "Page",
    "RagPipeline",
    "RetrievedChunk",
    "__version__",
]
