"""Exception hierarchy.

One base class so a caller can catch everything this package raises without catching
`Exception`, and narrow subclasses so the API layer can map failures onto status codes
without inspecting messages.
"""

from __future__ import annotations


class RagkitError(Exception):
    """Base for every error raised by this package."""


class ExtractionError(RagkitError):
    """A document could not be read, or produced no usable text."""


class UnsupportedFormatError(ExtractionError):
    """The file extension has no registered extractor."""


class EmbeddingError(RagkitError):
    """An embedding backend failed, or returned the wrong shape."""


class StoreError(RagkitError):
    """A vector store operation failed."""


class DimensionMismatchError(StoreError):
    """A vector's length does not match the collection it is being written to.

    Worth its own class because it almost always means the embedder was changed without
    re-indexing, which is a configuration mistake rather than a transient fault.
    """


class GenerationError(RagkitError):
    """The LLM call failed, or its response did not satisfy the response contract."""


class MalformedLLMResponseError(GenerationError):
    """The model returned something that is not the agreed JSON object.

    Separate from `GenerationError` because it is retryable in a way a transport failure is
    not: the same prompt may well parse on a second attempt.
    """
