#!/usr/bin/env python3
"""Measure retrieval and groundedness on a labelled set.

Why this exists: "95% retrieval accuracy" is a claim, and a claim without a metric, a `k`, a
query count and a labelling method is not checkable. This harness makes the numbers for *this*
repository reproducible — one command, no API key, no model download — and prints the
configuration alongside them, because a retrieval number without its chunk size and threshold
is not comparable to anything.

    python eval/run_eval.py
    python eval/run_eval.py --k 3 --chunk-chars 600 --json

Metrics
-------
**recall@k** — fraction of questions whose gold chunk appears in the top *k*. The headline
retrieval number. Chosen over precision because for RAG what matters is whether the answer is
*available* to the model, not how much irrelevant text came with it.

**MRR** — mean reciprocal rank of the gold chunk. Sensitive to *where* in the top *k* the
right passage landed, which recall@k cannot see; a system that always ranks gold 5th out of 5
has the same recall@5 as one that always ranks it 1st.

**grounded rate** — fraction of answerable questions that produced a verified, cited answer.
End-to-end, and the number that reflects the whole pipeline rather than retrieval alone.

**correct refusal rate** — fraction of *unanswerable* questions that were refused. Tracked
separately and deliberately: a system can score well on everything above by answering
everything, and that is the failure this project exists to prevent. The two must be read
together.

A note on the size of the set: 12 questions is a smoke test, not a benchmark. It is enough to
catch a regression and far too small to quote as an accuracy figure, which is why the output
prints the question count next to every percentage.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ragkit.config import Settings  # noqa: E402
from ragkit.embeddings import build_embedder  # noqa: E402
from ragkit.generate import ExtractiveStub  # noqa: E402
from ragkit.ground import normalise_for_match  # noqa: E402
from ragkit.pipeline import RagPipeline  # noqa: E402
from ragkit.prompts import PROMPT_VERSION  # noqa: E402
from ragkit.store import InMemoryStore  # noqa: E402

DATASET = Path(__file__).parent / "dataset.jsonl"
CORPUS = Path(__file__).parent / "corpus"


@dataclass
class Case:
    question: str
    answerable: bool
    gold_substring: str | None
    """Text that must appear in the gold chunk.

    Labelling by substring rather than by chunk id is deliberate: chunk ids shift whenever
    `chunk_chars` changes, so an id-labelled set could not be used to *compare* chunk sizes —
    which is one of the main things an eval harness is for.
    """


def load_cases() -> list[Case]:
    cases = []
    with DATASET.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            row = json.loads(line)
            cases.append(
                Case(
                    question=row["question"],
                    answerable=row["answerable"],
                    gold_substring=row.get("gold_substring"),
                )
            )
    return cases


def evaluate(settings: Settings) -> dict:
    embedder = build_embedder(
        settings.embedder,
        dimensions=settings.embedding_dimensions,
        model_name=settings.embedding_model,
    )
    pipeline = RagPipeline(
        embedder=embedder,
        store=InMemoryStore(),
        llm=ExtractiveStub(),
        settings=settings,
    )

    documents = sorted(CORPUS.glob("*.md")) + sorted(CORPUS.glob("*.txt"))
    if not documents:
        raise SystemExit(f"No corpus documents in {CORPUS}")
    for doc in documents:
        pipeline.ingest(doc)

    cases = load_cases()
    answerable = [c for c in cases if c.answerable]
    unanswerable = [c for c in cases if not c.answerable]

    hits = 0
    reciprocal_ranks: list[float] = []
    grounded = 0
    rows = []

    for case in answerable:
        vector = pipeline.embedder.embed_query(case.question)
        retrieved = pipeline.store.search(vector, limit=settings.top_k)

        rank = None
        assert case.gold_substring is not None
        # Whitespace-normalised comparison, via the same helper the verifier uses. A naive
        # `in` check failed three otherwise-correct retrievals purely because the source
        # document wraps lines mid-phrase ("a maximum of\nthree months"), which would have
        # been reported as a 79% recall score caused entirely by labelling.
        gold = normalise_for_match(case.gold_substring)
        for position, result in enumerate(retrieved, start=1):
            if gold in normalise_for_match(result.chunk.text):
                rank = position
                break
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        answer = pipeline.ask(case.question)
        if not answer.refused:
            grounded += 1

        rows.append(
            {
                "question": case.question,
                "gold_rank": rank,
                "grounded": not answer.refused,
                "verdict": answer.groundedness.verdict,
                "best_score": round(answer.groundedness.best_score, 4),
            }
        )

    correct_refusals = 0
    for case in unanswerable:
        answer = pipeline.ask(case.question)
        if answer.refused:
            correct_refusals += 1
        rows.append(
            {
                "question": case.question,
                "gold_rank": None,
                "grounded": not answer.refused,
                "verdict": answer.groundedness.verdict,
                "best_score": round(answer.groundedness.best_score, 4),
            }
        )

    n = len(answerable)
    return {
        "config": {
            "embedder": pipeline.embedder.name,
            "llm": pipeline.llm.model,
            "prompt_version": PROMPT_VERSION,
            "chunk_chars": settings.chunk_chars,
            "chunk_overlap_chars": settings.chunk_overlap_chars,
            "top_k": settings.top_k,
            "score_threshold": settings.score_threshold,
            "documents": len(documents),
            "chunks_indexed": pipeline.store.count(),
        },
        "counts": {
            "answerable": n,
            "unanswerable": len(unanswerable),
        },
        "metrics": {
            f"recall@{settings.top_k}": round(hits / n, 4) if n else 0.0,
            "mrr": round(sum(reciprocal_ranks) / n, 4) if n else 0.0,
            "grounded_rate": round(grounded / n, 4) if n else 0.0,
            "correct_refusal_rate": (
                round(correct_refusals / len(unanswerable), 4) if unanswerable else 0.0
            ),
        },
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--k", type=int, default=5, help="top_k for retrieval")
    parser.add_argument("--chunk-chars", type=int, default=700)
    parser.add_argument("--overlap-chars", type=int, default=120)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--embedder", default="hashing")
    parser.add_argument("--json", action="store_true", help="Emit the full result as JSON")
    args = parser.parse_args()

    settings = Settings(
        chunk_chars=args.chunk_chars,
        chunk_overlap_chars=args.overlap_chars,
        top_k=args.k,
        score_threshold=args.threshold,
        embedder=args.embedder,
        store="memory",
        llm="echo",
    ).validated()

    result = evaluate(settings)

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    config, counts, metrics = result["config"], result["counts"], result["metrics"]
    print("ragkit retrieval + groundedness eval")
    print("=" * 62)
    for key, value in config.items():
        print(f"  {key:22} {value}")
    print("-" * 62)
    for key, value in metrics.items():
        if key.startswith("recall") or key == "grounded_rate":
            print(f"  {key:22} {value:.1%}   (n={counts['answerable']})")
        elif key == "correct_refusal_rate":
            print(f"  {key:22} {value:.1%}   (n={counts['unanswerable']})")
        else:
            print(f"  {key:22} {value:.4f}")
    print("-" * 62)

    for row in result["cases"]:
        mark = (
            "ok  "
            if (row["gold_rank"] == 1 or row["verdict"] == "insufficient_retrieval")
            else "    "
        )
        rank = f"rank={row['gold_rank']}" if row["gold_rank"] else "rank=-"
        print(f"  {mark} {rank:8} {row['verdict']:24} {row['question'][:46]}")
    print("-" * 62)
    print(
        f"\n  {counts['answerable']} answerable + {counts['unanswerable']} unanswerable "
        f"questions. Too small to quote as an accuracy figure; sized to catch regressions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
