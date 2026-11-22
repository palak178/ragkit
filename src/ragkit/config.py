"""Configuration, from the environment with explicit defaults.

Every value a reviewer might want to change is here rather than inline, and every one is
documented with the trade-off it controls. `RAGKIT_` prefixed so it cannot collide — the
project this replaced read an environment variable literally named `api_key`.

Secrets are read from the environment and never written back to a model, a log line or an
API response. There is no `.env` in this repository; `.env.example` shows the shape.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAGKIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Chunking -----------------------------------------------------------------------
    #
    # Character-based, not token-based. Calling it token-aware would need a tokeniser per
    # model and would still only approximate the one actually used at generation time; the
    # honest description is that these are characters and the ratio to tokens is roughly 4:1
    # for English prose.
    chunk_chars: int = Field(
        default=1200,
        ge=200,
        le=8000,
        description="Target chunk size in characters. Larger means fewer, broader chunks: "
        "better recall of diffuse facts, worse precision and a bigger prompt.",
    )
    chunk_overlap_chars: int = Field(
        default=200,
        ge=0,
        description="Overlap between neighbours, so a fact spanning a boundary survives in "
        "at least one chunk whole. Must be smaller than chunk_chars.",
    )

    # -- Retrieval ----------------------------------------------------------------------
    top_k: int = Field(
        default=5, ge=1, le=50, description="Passages supplied to the model per question."
    )
    score_threshold: float = Field(
        default=0.10,
        ge=-1.0,
        le=1.0,
        description="Minimum top-1 cosine similarity before the model is called at all. "
        "Below this the question is refused without spending a token — the cheapest "
        "hallucination to prevent is the one where nothing relevant was found.",
    )
    """Default chosen by sweeping `eval/run_eval.py`, not by taste.

    On the 20-question eval set, with the hashing embedder:

        threshold   recall@5   grounded   correct refusal
        0.05        1.00       1.00       0.67
        0.08        1.00       1.00       1.00
        0.10        1.00       0.93       1.00
        0.15        1.00       0.79       1.00
        0.25        1.00       0.21       1.00

    0.08 scores perfectly on that set, which is exactly why it is not the default: a value
    sitting on the edge of a 20-question sample is fitted to the sample. 0.10 keeps every
    correct refusal, costs one answerable question out of fourteen, and leaves margin — the
    right direction for a project whose argument is that a refusal beats an unsupported
    answer.

    It is also specific to the lexical backend. Transformer similarities are compressed into
    a higher, narrower band, so switching to `sentence-transformers` means re-running the
    sweep rather than inheriting this number.
    """

    # -- Embeddings ---------------------------------------------------------------------
    embedder: str = Field(
        default="hashing",
        description="`hashing` (built in, offline, lexical) or `sentence-transformers` "
        "(needs the `local` extra, semantic).",
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Only read when embedder=sentence-transformers.",
    )
    embedding_dimensions: int = Field(
        default=512,
        ge=64,
        description="Only read when embedder=hashing; the hashing output width.",
    )

    # -- Vector store -------------------------------------------------------------------
    store: str = Field(default="memory", description="`memory` or `qdrant`.")
    qdrant_url: str | None = Field(default=None, description="e.g. http://localhost:6333")
    qdrant_api_key: str | None = Field(default=None, repr=False)
    qdrant_collection: str = Field(default="ragkit")

    # -- Generation ---------------------------------------------------------------------
    llm: str = Field(default="echo", description="`echo` (offline stub), `openai` or `anthropic`.")
    llm_model: str = Field(default="gpt-4o-mini")
    openai_api_key: str | None = Field(default=None, repr=False)
    anthropic_api_key: str | None = Field(default=None, repr=False)
    max_answer_tokens: int = Field(default=700, ge=64)

    def validated(self) -> Settings:
        """Cross-field checks that a single field validator cannot express."""
        if self.chunk_overlap_chars >= self.chunk_chars:
            raise ValueError(
                f"chunk_overlap_chars ({self.chunk_overlap_chars}) must be smaller than "
                f"chunk_chars ({self.chunk_chars}); otherwise chunking cannot advance."
            )
        if self.store == "qdrant" and not self.qdrant_url:
            raise ValueError("store=qdrant needs RAGKIT_QDRANT_URL")
        return self


def load_settings() -> Settings:
    return Settings().validated()
