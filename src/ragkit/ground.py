"""The deterministic half.

This module is the reason the project exists. The model proposes an answer and the evidence
it claims to have used; nothing here trusts either. Every decision below is ordinary code
operating on strings:

* **Before** generation, retrieval is gated on a score threshold. If the best passage is not
  similar enough, the question is refused without calling the model at all — the cheapest
  hallucination to prevent is the one about a corpus that contains nothing relevant.
* **After** generation, every quote the model offered is looked for in the passage it
  attributed it to. A quote that is not there is not evidence, however plausible it reads.
* Every inline `[n]` in the answer must resolve to a passage whose quote verified.

The last rule is strict on purpose: one unverifiable citation refuses the whole answer rather
than publishing the rest with a gap. That trades recall for precision, which is the right
direction when the output is meant to survive review — an answer nobody can audit is worth
less than an honest "not found". The counts in `GroundednessReport` make the trade visible
instead of hiding it.
"""

from __future__ import annotations

import re
import unicodedata

from ragkit.models import Citation, GroundednessReport, RetrievedChunk

# Inline citation markers: [1], [12]. Non-numeric brackets are left alone, so prose that
# happens to contain [sic] is not mistaken for a citation.
_MARKER = re.compile(r"\[(\d{1,3})\]")

# Models routinely "improve" punctuation when copying a quote — curly quotes for straight
# ones, an en dash for a hyphen, a non-breaking space. Those substitutions are cosmetic but
# they break a naive substring check, which would then report a faithful quote as unverified.
# Folding them is the difference between checking meaning and checking typography.
_PUNCTUATION_FOLD = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
        " ": " ",
        " ": " ",
        " ": " ",
        " ": " ",
        " ": " ",
        "…": "...",
    }
)

MIN_QUOTE_CHARS = 12
"""Shorter spans are rejected as evidence.

A three-word quote appears in almost any passage by chance, so accepting one would make the
verification step a formality. Twelve characters is long enough that a match means something
and short enough to allow a genuine short fact ("30 days' notice").
"""


def normalise_for_match(text: str) -> str:
    """Canonical form for substring comparison.

    NFKC first (so a ligature or a full-width digit compares equal to its plain form), then
    punctuation folding, then case folding, then whitespace collapse. Applied identically to
    both sides, so it can only ever make a true match findable — never invent one, since no
    step removes or reorders words.
    """
    folded = unicodedata.normalize("NFKC", text).translate(_PUNCTUATION_FOLD)
    return " ".join(folded.casefold().split())


def inline_markers(answer: str) -> list[int]:
    """Passage numbers cited inline, in order of first appearance, without duplicates."""
    seen: dict[int, None] = {}
    for match in _MARKER.finditer(answer):
        seen.setdefault(int(match.group(1)), None)
    return list(seen)


def gate_retrieval(
    retrieved: list[RetrievedChunk], *, threshold: float
) -> GroundednessReport | None:
    """Decide whether generation is worth attempting.

    Returns a refusing report, or `None` to mean "proceed". Returning the report rather than a
    bare bool keeps the reason for the refusal in one place.
    """
    if not retrieved:
        return GroundednessReport(
            verdict="insufficient_retrieval",
            best_score=-1.0,
            score_threshold=threshold,
            chunks_retrieved=0,
        )
    best = retrieved[0].score
    if best < threshold:
        return GroundednessReport(
            verdict="insufficient_retrieval",
            best_score=best,
            score_threshold=threshold,
            chunks_retrieved=len(retrieved),
        )
    return None


def verify(
    *,
    answer: str,
    support: list[dict],
    retrieved: list[RetrievedChunk],
    threshold: float,
) -> tuple[GroundednessReport, list[Citation]]:
    """Check a model's answer against the passages it was given.

    `support` is the raw list from the model: dicts with `passage` and `quote`. Raw on
    purpose — this function is the boundary where untrusted output becomes typed `Citation`
    objects, so it must cope with whatever arrived.
    """
    best = retrieved[0].score if retrieved else -1.0
    # Passage numbers are 1-based in the prompt, because that is how the model is asked to
    # cite them and off-by-one here would silently mis-attribute every citation.
    by_number = dict(enumerate(retrieved, start=1))

    out_of_range: list[int] = []
    quote_not_found: list[int] = []
    citations: list[Citation] = []
    verified_passages: set[int] = set()

    for entry in support:
        number = entry.get("passage")
        quote = (entry.get("quote") or "").strip()
        if not isinstance(number, int) or number not in by_number:
            # Includes the common failure where a model cites a passage index it was never
            # shown, which is a fabricated source rather than a formatting slip.
            if isinstance(number, int):
                out_of_range.append(number)
            continue
        if len(normalise_for_match(quote)) < MIN_QUOTE_CHARS:
            quote_not_found.append(number)
            continue

        chunk = by_number[number].chunk
        if normalise_for_match(quote) not in normalise_for_match(chunk.text):
            quote_not_found.append(number)
            continue

        if number in verified_passages:
            continue  # One citation per passage is enough; duplicates add no evidence.
        verified_passages.add(number)
        citations.append(
            Citation(
                marker=number,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                page=chunk.page,
                quote=quote,
            )
        )

    cited_inline = inline_markers(answer)
    unsupported_inline = [n for n in cited_inline if n not in verified_passages]

    grounded = bool(answer.strip()) and bool(citations) and not unsupported_inline
    report = GroundednessReport(
        verdict="grounded" if grounded else "ungrounded",
        best_score=best,
        score_threshold=threshold,
        chunks_retrieved=len(retrieved),
        support_claimed=len(support),
        support_verified=len(citations),
        rejected_out_of_range=sorted(set(out_of_range) | set(unsupported_inline)),
        rejected_quote_not_found=sorted(set(quote_not_found)),
    )
    # Markers are kept as the passage numbers the model used, so [2] in the answer and
    # `marker=2` in the citation list refer to the same passage with no remapping to get
    # wrong. Citations are ordered by first appearance for readability.
    order = {n: i for i, n in enumerate(cited_inline)}
    citations.sort(key=lambda c: (order.get(c.marker, len(order)), c.marker))
    return report, (citations if grounded else [])


def refusal_text(report: GroundednessReport) -> str:
    """A plain-language reason, suitable for showing a user."""
    if report.verdict == "insufficient_retrieval":
        if report.chunks_retrieved == 0:
            return "Nothing is indexed yet, so there is no source material to answer from."
        return (
            f"The closest passage scored {report.best_score:.3f}, below the "
            f"{report.score_threshold:.3f} threshold, so the corpus does not appear to cover "
            f"this question. No answer was generated."
        )
    if report.verdict == "model_declined":
        return "The retrieved passages do not contain the answer."
    if report.verdict == "ungrounded":
        parts = []
        if report.rejected_quote_not_found:
            parts.append(
                f"quotes for passage(s) {report.rejected_quote_not_found} were not found in "
                f"the cited text"
            )
        if report.rejected_out_of_range:
            parts.append(f"passage(s) {report.rejected_out_of_range} were cited without evidence")
        if not report.support_claimed:
            parts.append("no supporting quotes were offered")
        detail = "; ".join(parts) or "the answer could not be tied to the sources"
        return f"A draft answer failed verification and was withheld: {detail}."
    return ""
