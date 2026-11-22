"""Embedding backends.

Two of them, and the default is the honest one rather than the impressive one.

`HashingEmbedder` is a hashed bag-of-words — the classic hashing-trick vectoriser. It is
**lexical, not semantic**: it matches shared wording, not shared meaning, so it will miss a
paraphrase that a transformer catches. It is the default because it makes a fresh clone
work offline, in milliseconds, with no model download, no torch and no API key, which means
the test suite and the eval harness measure this package's logic rather than a model's. For
real semantic retrieval, install the `local` extra and set `RAGKIT_EMBEDDER=sentence-transformers`.

Calling the hashing backend "semantic search" would be an overclaim, so neither this module
nor the README does.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from functools import lru_cache
from itertools import pairwise

from ragkit.errors import EmbeddingError

_TOKEN = re.compile(r"[a-z0-9]+")

# Function words are removed before hashing. This is not cosmetic tuning — without it the
# backend is actively misleading. "What is the capital of Mongolia?" against an HR policy
# shares `is`, `the`, `of` and nothing else, which was enough to score 0.198 and clear the
# relevance gate, so a question the corpus could not possibly answer got answered. Content
# words only, and that query now scores 0.0 and is refused.
#
# A transformer does not need this; it is a cost of the lexical baseline, which is why it
# lives here rather than in the shared tokeniser.
STOPWORDS = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "cannot",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "must",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    ]
)


def tokenise(text: str) -> list[str]:
    """Lower-case alphanumeric tokens. Deliberately simple and language-agnostic."""
    return _TOKEN.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens with function words removed, preserving order."""
    return [t for t in tokenise(text) if t not in STOPWORDS]


@lru_cache(maxsize=100_000)
def _bucket(token: str, dimensions: int) -> tuple[int, float]:
    """Map a token to `(index, sign)`.

    Uses blake2b rather than `hash()`: Python salts string hashing per process, so `hash()`
    would give a different index on every run and an index written today would not be
    searchable tomorrow. That is the kind of bug that looks like bad retrieval quality.

    The signed hashing trick (Weinberger et al.) lets collisions cancel on average instead of
    always inflating a bucket.
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dimensions, 1.0 if (value >> 63) & 1 else -1.0


class HashingEmbedder:
    """Deterministic, offline, lexical embeddings.

    Unigrams and bigrams: bigrams cost one extra pass and recover most of the phrase
    sensitivity a pure bag of words throws away ("interest rate" versus "rate of interest").
    Term frequency is damped with `1 + log(tf)` so a word repeated twenty times does not
    dominate, and the vector is L2-normalised so a cosine similarity is a plain dot product.
    """

    def __init__(self, dimensions: int = 512) -> None:
        if dimensions < 64:
            raise EmbeddingError(f"dimensions must be >= 64, got {dimensions}")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def name(self) -> str:
        return f"hashing-{self._dimensions}"

    def _vector(self, text: str) -> list[float]:
        # Bigrams are built from the *filtered* stream, so "twenty-one days of annual leave"
        # yields `days_annual` and `annual_leave` rather than `of_annual`. Pairing content
        # words across removed function words is what keeps phrase sensitivity useful.
        tokens = content_tokens(text)
        if not tokens:
            # An all-zero vector has no direction, so cosine similarity against it is
            # undefined. Returning it explicitly (rather than dividing by zero) lets the
            # store score it as 0.0 and the threshold reject it, which is the right outcome
            # for an empty query.
            return [0.0] * self._dimensions

        grams = Counter(tokens)
        grams.update(f"{a}_{b}" for a, b in pairwise(tokens))

        vector = [0.0] * self._dimensions
        for gram, count in grams.items():
            index, sign = _bucket(gram, self._dimensions)
            vector[index] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class SentenceTransformerEmbedder:
    """Real semantic embeddings via sentence-transformers. Needs the `local` extra.

    The model is loaded once, lazily, on first use rather than in `__init__`, so constructing
    the object in a test or a CLI `--help` path does not pay for a model download.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None

    @property
    def name(self) -> str:
        return self._model_name

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed. `pip install '.[local]'`, or set "
                    "RAGKIT_EMBEDDER=hashing to stay offline."
                ) from exc
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dimensions(self) -> int:
        dim = self._load().get_sentence_embedding_dimension()
        if dim is None:  # pragma: no cover - defensive
            raise EmbeddingError(f"{self._model_name} did not report its dimensionality")
        return int(dim)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load().encode(texts, normalize_embeddings=True)
        return [[float(x) for x in row] for row in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._load().encode([text], normalize_embeddings=True)[0]
        return [float(x) for x in vector]


def build_embedder(kind: str, *, dimensions: int = 512, model_name: str = "") -> object:
    """Resolve the configured embedder name to an instance."""
    if kind == "hashing":
        return HashingEmbedder(dimensions=dimensions)
    if kind in {"sentence-transformers", "sentence_transformers", "st"}:
        return SentenceTransformerEmbedder(model_name or "sentence-transformers/all-MiniLM-L6-v2")
    raise EmbeddingError(f"Unknown embedder '{kind}'. Use 'hashing' or 'sentence-transformers'.")
