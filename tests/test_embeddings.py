"""Tests for the embedding backends.

The determinism test is the important one. `hash()` is salted per process in Python, so an
embedder built on it produces different vectors in every run — an index written today would
not be searchable tomorrow, and the symptom is "retrieval quality is bad" rather than an
error. Asserting stability across a *subprocess* is the only way to catch that.
"""

from __future__ import annotations

import math
import subprocess
import sys

import pytest

from ragkit.embeddings import (
    HashingEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
    content_tokens,
    tokenise,
)
from ragkit.errors import EmbeddingError
from ragkit.store import cosine_similarity


class TestTokenisation:
    def test_lowercases_and_splits_on_punctuation(self):
        assert tokenise("Thirty-Days, written NOTICE!") == ["thirty", "days", "written", "notice"]

    def test_content_tokens_drop_function_words(self):
        assert content_tokens("the notice period is for the employee") == [
            "notice",
            "period",
            "employee",
        ]

    def test_content_tokens_preserve_order(self):
        assert content_tokens("alpha and beta and gamma") == ["alpha", "beta", "gamma"]


class TestHashingEmbedder:
    def test_dimensions_and_name(self):
        e = HashingEmbedder(dimensions=256)
        assert e.dimensions == 256
        assert e.name == "hashing-256"
        assert len(e.embed_query("anything at all")) == 256

    def test_rejects_tiny_dimensions(self):
        with pytest.raises(EmbeddingError, match=">= 64"):
            HashingEmbedder(dimensions=8)

    def test_vectors_are_l2_normalised(self):
        v = HashingEmbedder().embed_query("notice period for resigning employees")
        assert math.sqrt(sum(x * x for x in v)) == pytest.approx(1.0, abs=1e-9)

    def test_identical_text_is_identical_vector(self):
        e = HashingEmbedder()
        assert e.embed_query("notice period") == e.embed_query("notice period")

    def test_documents_and_query_agree(self):
        # The hashing backend is symmetric; if that ever stops being true, retrieval silently
        # degrades, so it is asserted rather than assumed.
        e = HashingEmbedder()
        assert e.embed_documents(["notice period"])[0] == e.embed_query("notice period")

    def test_empty_and_stopword_only_text_gives_a_zero_vector(self):
        e = HashingEmbedder()
        for text in ["", "   ", "the of and is"]:
            assert e.embed_query(text) == [0.0] * e.dimensions

    def test_zero_vector_scores_zero_rather_than_raising(self):
        e = HashingEmbedder()
        assert cosine_similarity(e.embed_query("the of and"), e.embed_query("notice")) == 0.0

    def test_related_text_scores_above_unrelated(self):
        e = HashingEmbedder()
        source = e.embed_documents(
            ["Employees must give thirty days written notice before resigning."]
        )[0]
        near = cosine_similarity(e.embed_query("how much notice before resigning"), source)
        far = cosine_similarity(e.embed_query("capital city of Mongolia"), source)
        assert near > far
        assert far == pytest.approx(0.0, abs=1e-9)

    def test_bigrams_make_word_order_matter(self):
        # A pure bag of words scores these identically. Bigrams are what separate them.
        e = HashingEmbedder()
        a = e.embed_documents(["interest rate review"])[0]
        assert cosine_similarity(e.embed_query("interest rate review"), a) > cosine_similarity(
            e.embed_query("review rate interest"), a
        )

    def test_repeated_terms_are_damped(self):
        # Sublinear term frequency: twenty repeats must not dominate the vector.
        e = HashingEmbedder()
        once = e.embed_documents(["notice policy"])[0]
        many = e.embed_documents(["notice " * 20 + "policy"])[0]
        assert cosine_similarity(once, many) > 0.5

    def test_embed_documents_preserves_order_and_length(self):
        e = HashingEmbedder()
        out = e.embed_documents(["alpha one", "beta two", "gamma three"])
        assert len(out) == 3
        assert out[0] == e.embed_query("alpha one")

    def test_empty_batch(self):
        assert HashingEmbedder().embed_documents([]) == []

    def test_vectors_are_stable_across_processes(self):
        """The regression test for using `hash()` instead of a real digest."""
        code = (
            "from ragkit.embeddings import HashingEmbedder;"
            "print(HashingEmbedder(128).embed_query('notice period for employees')[:6])"
        )
        runs = {
            subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, check=True
            ).stdout.strip()
            for _ in range(2)
        }
        assert len(runs) == 1, "vectors differ between processes; hashing is not deterministic"
        assert runs.pop() == str(
            HashingEmbedder(128).embed_query("notice period for employees")[:6]
        )


class TestFactory:
    def test_builds_hashing(self):
        e = build_embedder("hashing", dimensions=128)
        assert isinstance(e, HashingEmbedder)
        assert e.dimensions == 128

    @pytest.mark.parametrize("alias", ["sentence-transformers", "sentence_transformers", "st"])
    def test_builds_sentence_transformers_without_loading_a_model(self, alias):
        # Construction must stay lazy, or `--help` and every test would download weights.
        assert isinstance(build_embedder(alias, model_name="x/y"), SentenceTransformerEmbedder)

    def test_unknown_name_is_an_error(self):
        with pytest.raises(EmbeddingError, match="Unknown embedder"):
            build_embedder("word2vec")

    def test_sentence_transformers_reports_a_clear_error_when_missing(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fail(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail)
        with pytest.raises(EmbeddingError, match="local"):
            SentenceTransformerEmbedder("x/y").embed_query("text")
