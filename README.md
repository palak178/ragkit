# ragkit

**Grounded retrieval-augmented question answering: a cited answer, or an explicit refusal.
Never an uncited one.**

[![CI](https://github.com/palak178/LLM_Chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/palak178/LLM_Chatbot/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)

Most RAG examples retrieve some passages, paste them into a prompt, and return whatever the
model says. That works until the model says something the passages do not support — and then
nothing in the system notices, because nothing in the system was ever checking.

ragkit checks. The model proposes an answer **and the evidence it used**; deterministic code
then looks for each quote in the passage it was attributed to. A quote that is not there is
not evidence, however plausible it reads, and the answer is withheld.

> The LLM plans; deterministic code decides.

---

## Contents

- [Quickstart](#quickstart) · [What makes an answer publishable](#what-makes-an-answer-publishable)
- [Architecture](#architecture) · [HTTP API](#http-api) · [Configuration](#configuration)
- [Evaluation](#evaluation) · [Design decisions](#design-decisions)
- [What is deliberately not built](#what-is-deliberately-not-built) · [Testing](#testing)

---

## Quickstart

No API key, no model download, no database. A fresh clone works offline.

```bash
git clone https://github.com/palak178/LLM_Chatbot.git ragkit && cd ragkit
python -m pip install -e ".[dev]"

make demo          # a question the corpus answers
make demo-refusal  # one it does not
```

```console
$ ragkit ask "How much notice must an employee give before resigning?" \
      --doc eval/corpus/employee_handbook.md

Employees must give thirty days written notice before resigning. [1]

[1] employee_handbook.md
  [1] "Employees must give thirty days written notice before resigning."
```

```console
$ ragkit ask "What is the capital of Mongolia?" \
      --doc eval/corpus/employee_handbook.md

The closest passage scored 0.000, below the 0.100 threshold, so the corpus does not appear
to cover this question. No answer was generated.
  verdict=insufficient_retrieval best_score=0.000 threshold=0.100
```

The second case never reaches the model. Exit codes differ too — `0` answered, `3` refused,
`2` misuse — so this is usable in a script.

In Python:

```python
from ragkit import RagPipeline

pipeline = RagPipeline.from_settings()
pipeline.ingest("handbook.pdf")

answer = pipeline.ask("How much notice is required?")
if answer.refused:
    print(answer.refusal_reason)
else:
    print(answer.text)
    for c in answer.citations:
        print(f"[{c.marker}] {c.doc_id} p.{c.page}: {c.quote}")
```

> **The default backends are honest, not impressive.** Out of the box you get *lexical*
> hashing embeddings and an *extractive* stub that is not a language model. Both are chosen so
> the tests and the eval measure this package rather than a vendor's model. See
> [Design decisions](#design-decisions) for how to switch to real ones.

---

## What makes an answer publishable

Two gates, both deterministic. Neither asks a model to grade itself.

### Gate 1 — retrieval, before any token is spent

If the best passage scores below `score_threshold`, the question is refused **without calling
the model at all**. The cheapest hallucination to prevent is the one about a corpus that
contains nothing relevant, and a test asserts the model was never invoked:

```python
assert llm.calls == [], "the model was called despite failing the retrieval gate"
```

### Gate 2 — verification, on what the model actually produced

The model must reply with a JSON object naming the passage and quoting the exact span it
relied on:

```json
{"sufficient": true,
 "answer": "Employees must give thirty days written notice. [1]",
 "support": [{"passage": 1, "quote": "thirty days written notice"}]}
```

Then, in ordinary code:

| Check | Failure recorded as |
|---|---|
| Does the quote appear in the passage it cites? | `rejected_quote_not_found` |
| Is the cited passage one that was actually supplied? | `rejected_out_of_range` |
| Is the quote long enough to be evidence (≥ 12 chars)? | `rejected_quote_not_found` |
| Does every inline `[n]` resolve to a verified quote? | `rejected_out_of_range` |
| Is there at least one verified quote, and non-empty text? | `ungrounded` |

Comparison is whitespace-, case- and punctuation-normalised, so a model that retypes a quote
with curly quotes or an en dash is not accused of fabricating it. Normalisation never
reorders or drops words, so it can only make a *true* match findable — never invent one.

**One unverifiable citation refuses the whole answer.** That is strict on purpose: publishing
the verified half with a dangling citation is the exact failure being prevented. It trades
recall for precision, which is the right direction when the output is meant to survive
review — and `GroundednessReport` publishes the counts so the trade is visible rather than
hidden.

Every refusal says why, in numbers:

```json
{"verdict": "ungrounded", "best_score": 0.41, "score_threshold": 0.10,
 "chunks_retrieved": 5, "support_claimed": 2, "support_verified": 1,
 "rejected_out_of_range": [2], "rejected_quote_not_found": []}
```

---

## Architecture

```
          ingest                                  ask
            │                                      │
      ┌─────▼─────┐                          ┌─────▼─────┐
      │  extract  │ pypdf / utf-8            │  embed    │
      └─────┬─────┘ loud on a scan           └─────┬─────┘
            │                                      │
      ┌─────▼─────┐                          ┌─────▼─────┐
      │  chunk    │ overlapping, snapped,    │  search   │ cosine, exact
      └─────┬─────┘ page preserved           └─────┬─────┘
            │                                      │
      ┌─────▼─────┐                     ╔══════════▼════════╗  below
      │  embed    │                     ║ GATE 1            ║  threshold
      └─────┬─────┘                     ║ retrieval score   ║──────────┐
            │                           ╚══════════┬════════╝          │
      ┌─────▼─────┐                                │                   │
      │  upsert   │ text in payload,         ┌─────▼─────┐             │
      │           │ keyed on chunk_id        │ generate  │ JSON        │
      └───────────┘                          └─────┬─────┘ contract    │
                                                   │                   │
                                        ╔══════════▼════════╗ quote    │
                                        ║ GATE 2            ║ not      │
                                        ║ quote in source?  ║ found    │
                                        ╚══════════┬════════╝──────────┤
                                                   │                   ▼
                                           cited answer        refusal + report
```

One module per job, and the two that matter are small enough to read in full:

| Module | Responsibility |
|---|---|
| `extract.py` | PDF and text → pages. Raises rather than indexing nothing |
| `chunking.py` | Overlapping chunks spanning pages, each carrying its source page |
| `embeddings.py` | `HashingEmbedder` (offline, lexical) · `SentenceTransformerEmbedder` |
| `store.py` | `InMemoryStore` (exact) · `QdrantStore` (payload-bearing, idempotent) |
| `generate.py` | Prompt assembly, response parsing, three LLM backends |
| **`ground.py`** | **Both gates. No model involvement. The reason the project exists** |
| `pipeline.py` | Sequences the above; owns the retry policy |
| `ports.py` | `Embedder`, `VectorStore`, `LLM` as `Protocol`s |

`ports.py` is why installing this package pulls no torch, no vector database and no provider
SDK, and why the whole suite runs offline against plain-class fakes.

---

## HTTP API

```bash
pip install -e ".[api]"
ragkit serve     # 127.0.0.1:8000, localhost by default
```

| Endpoint | Behaviour |
|---|---|
| `GET /health` | Status, version, chunks indexed, embedder and LLM in use |
| `POST /ingest` | Multipart upload → `201` + `IngestReport`. `415` unsupported type, `422` unreadable |
| `POST /ask` | `{"question": "..."}` → `200` + `Answer` |

**A refusal is `200` with `"text": null`.** A question the corpus cannot answer is a correct
outcome of a working system; returning 4xx or 5xx would teach clients to treat "I don't know"
as a fault to retry around. The groundedness report is always included, so a UI can explain
*why* rather than just *that*.

Uploads take only the final path component of `filename`, which is attacker-controlled —
`../../etc/passwd.md` is ingested as `passwd.md`, and there is a test for it.

---

## Configuration

Environment variables, `RAGKIT_` prefixed. See [`.env.example`](.env.example); `.env` is
git-ignored.

| Variable | Default | Notes |
|---|---|---|
| `RAGKIT_CHUNK_CHARS` | `1200` | **Characters, not tokens** (~4:1 for English prose) |
| `RAGKIT_CHUNK_OVERLAP_CHARS` | `200` | Must be smaller than `CHUNK_CHARS`; rejected at startup |
| `RAGKIT_TOP_K` | `5` | Passages supplied per question |
| `RAGKIT_SCORE_THRESHOLD` | `0.10` | Gate 1. Chosen by sweep, not taste — see below |
| `RAGKIT_EMBEDDER` | `hashing` | or `sentence-transformers` (needs `[local]`) |
| `RAGKIT_STORE` | `memory` | or `qdrant` (needs `[qdrant]`) |
| `RAGKIT_LLM` | `echo` | the extractive stub; or `openai` / `anthropic` |

Cross-field validation happens at startup (`Settings.validated()`), so an overlap wider than
the chunk size is a clear error rather than a hang.

---

## Evaluation

"95% retrieval accuracy" is not a checkable claim without a metric, a *k*, a query count and a
labelling method. So:

```bash
make eval                               # or: python eval/run_eval.py --json
python eval/run_eval.py --threshold 0.2 --chunk-chars 400
```

```
  embedder               hashing-512
  llm                    extractive-stub
  prompt_version         2026-11-23.1
  chunk_chars            700
  top_k                  5
  score_threshold        0.10
--------------------------------------------------------------
  recall@5               100.0%   (n=14)
  mrr                    0.9643
  grounded_rate          92.9%    (n=14)
  correct_refusal_rate   100.0%   (n=6)
```

**Read the last two lines together.** A system can score perfectly on the first three by
answering everything, which is the failure this project exists to prevent — so unanswerable
questions are a scored category, not an afterthought.

The threshold default came from sweeping it:

| threshold | recall@5 | grounded | correct refusal |
|---|---|---|---|
| 0.05 | 1.00 | 1.00 | 0.67 |
| 0.08 | 1.00 | 1.00 | 1.00 |
| **0.10** ← default | 1.00 | 0.93 | 1.00 |
| 0.15 | 1.00 | 0.79 | 1.00 |
| 0.25 | 1.00 | 0.21 | 1.00 |

0.08 scores perfectly, which is precisely why it is not the default: a value sitting on the
edge of a 20-question sample is fitted to the sample. 0.10 keeps every correct refusal, costs
one answerable question in fourteen, and leaves margin.

**Honest caveat: 20 questions is a regression test, not a benchmark.** The output prints *n*
beside every percentage for that reason. These numbers say "retrieval and refusal still
behave"; they are not an accuracy figure for RAG in general. The gold labels are matched by
normalised substring — a method that itself caused three false misses until it was fixed to
use the same comparison the verifier uses.

---

## Design decisions

**Chunking spans pages rather than restarting at each one.** Per-page chunking keeps
provenance trivially exact but fragments every sentence crossing a page break and emits a
useless 40-character chunk for every short page. Pages are joined, the join is chunked, and
each chunk's page is recovered from its start offset. Blank pages contribute no offsets, so
they cannot shift later citations onto the wrong page.

**The default embedder is lexical, and this README says so.** `HashingEmbedder` is a signed
hashing-trick bag of words with bigrams and damped term frequency. It matches shared *wording*,
not shared *meaning* — it will miss a paraphrase a transformer catches. It is the default
because it makes a clone work offline in milliseconds, which is what lets the eval measure
chunking and verification instead of a vendor's model. Calling it "semantic search" would be
the same category of overclaim this repository was rewritten to remove.

It uses **blake2b, not `hash()`**. Python salts string hashing per process, so a `hash()`-based
embedder writes different vectors on every run: an index built today is not searchable
tomorrow, and the symptom is "retrieval quality is bad" rather than an error.
`test_vectors_are_stable_across_processes` spawns a subprocess to prove it.

**Function words are filtered, and that is not cosmetic.** Without it, *"What is the capital of
Mongolia?"* against an HR handbook shares `is`, `the`, `of` — enough to score **0.198**, clear
the gate, and get answered. With content words only it scores **0.000** and is refused. A
transformer does not need this; it is a cost of the lexical backend, and the eval found it
rather than inspection.

**The default LLM is not an LLM.** `ExtractiveStub` picks the passage with the most content-word
overlap, quotes its best sentence verbatim, and cites it. Its answers are copied, not written,
so they read stiffly and cannot synthesise across passages. It exists because a project that
only runs with somebody's API key cannot be tested, evaluated or reviewed by a stranger. For
real prose set `RAGKIT_LLM=openai`.

**One retry on an unparseable response, then refuse.** A malformed reply is usually a sampling
accident and often parses on a second attempt. More retries throw money at a prompt that is
not working and hide it from whoever should fix it. A model that will not produce the contract
is reported as "no grounded answer", not as a 500 — from a caller's view it is
indistinguishable from a model that cannot answer.

**Exact brute-force search, not an approximate index.** For a few thousand chunks an
exhaustive scan is milliseconds and returns *exact* neighbours, so retrieval quality measured
against it is the pipeline's rather than an index's recall curve. Approximate indexes earn
their complexity around a million vectors; `QdrantStore` is there for that.

---

## What is deliberately not built

Each of these is a real gap, listed because a reviewer who finds one unlisted is right to
wonder what else is missing.

| Not built | Why |
|---|---|
| **Reranking** | The highest-value next addition. A cross-encoder over the top ~50 reliably beats pure vector ranking. It would be a fourth `Protocol` in `ports.py`; there is no stub pretending otherwise |
| **Hybrid (BM25 + vector) retrieval** | The hashing embedder is already lexical, so hybrid search would largely duplicate it. It becomes worthwhile *with* a transformer embedder, and belongs with that change |
| **OCR** | A scanned PDF raises `ExtractionError` naming OCR rather than indexing nothing. Wiring Tesseract is easy; being honest that the text is pixels matters more |
| **Auth, multi-tenancy, rate limits** | The API binds to localhost and has no identity. Tenant scoping is a metadata filter plus an index-per-tenant decision, which is meaningless without the authentication that would populate the tenant id |
| **Conversational memory** | Single-turn only. Follow-ups ("what about part-timers?") need query rewriting against history, which is its own correctness problem and not yet addressed |
| **Streaming responses** | The verification gate is inherently non-streaming: you cannot check a citation in a sentence that has not finished. Streaming would mean publishing unverified text and retracting it |
| **Sentence-level attribution** | Citations attach to the answer, not to each sentence. Per-sentence grounding needs claim splitting and an entailment check, not a substring test |

---

## Testing

```bash
make test        # 191 tests, offline, ~0.4s
make test-all    # plus Qdrant integration (needs a server)
make lint
```

Both `pytest` and `python -m pytest` work — they resolve `sys.path` differently, and CI runs
both so a fresh clone cannot hit a collection error the Makefile hides.

What the suite is built around:

- **`tests/test_ground.py` is the centre.** Fabricated quotes, citations to unsupplied
  passages, quotes too short to be evidence, inline markers without support, and the
  malformed support shapes a real model actually emits.
- **`ScriptedLLM` makes fabrication reachable.** A real model cannot be made to invent a quote
  on demand, so a scripted one does — and the assertions run *through the pipeline*, where a
  wiring mistake would hide, not only against `ground.verify` in isolation.
- **The PDF test uses a real PDF**, assembled byte by byte in `conftest.minimal_pdf` with a
  correct xref table. A mocked `PdfReader` would pass whether or not PDF extraction works,
  which is exactly the bug this repository used to have.
- **No mocked vector database.** `QdrantStore` is tested against a real server in CI and skips
  locally without one. A mock would assert only that the mock was called.

---

## History

This repository began in 2023 as a Streamlit demo and was rewritten in November 2026. The
rewrite was not cosmetic: the original claimed "retrieval-grounded answers over uploaded PDFs"
and did none of it. It built a prompt containing the retrieved context and then sent a
*different* variable to the model, so the context was computed and discarded; it stored points
with no payload, so the index could not return text at all; it wrote every point with a
hard-coded `id=1`, so each upload destroyed the last; and it decoded PDF bytes as latin-1
instead of parsing them.

Those four bugs are why several modules here carry a comment explaining what they are *not*
doing, and why `test_store.py` and `test_qdrant_integration.py` both assert that the payload
round-trips. A claim in a README should be something a test can fail on.

The pre-rewrite history also contained a live API key. It has been purged and the credentials
rotated — but a key that was once public is compromised regardless, which is the whole
argument for `.env.example` and for the comment at the top of `.gitignore`.

## Licence

MIT — see [LICENSE](LICENSE).
