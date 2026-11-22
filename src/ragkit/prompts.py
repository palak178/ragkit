"""Prompts, versioned.

Kept in one module with an explicit `PROMPT_VERSION` because a prompt is as much a part of
behaviour as code: when an eval number moves, the first question is which prompt produced it.
The version is recorded in eval output for exactly that reason.

The response contract is JSON rather than prose. Not for tidiness — it is what lets
`ground.py` check the model's work. Free-text "according to passage 2" cannot be verified
mechanically; `{"passage": 2, "quote": "..."}` can be, by looking for that quote in that
passage. The model is asked to supply the evidence it used, and then deterministic code
decides whether the evidence actually says so.
"""

from __future__ import annotations

PROMPT_VERSION = "2026-11-23.1"

SYSTEM = """\
You answer questions strictly from numbered passages supplied by the user. You are part of a \
pipeline that mechanically verifies your output, so accuracy matters more than helpfulness.

Rules:
1. Use only the passages. Do not use prior knowledge, and do not infer beyond what is written.
2. Cite inline with the passage number in square brackets, like [2], after each claim.
3. For every passage you cite, supply a `support` entry quoting the exact span you relied on, \
copied verbatim from that passage. Do not paraphrase inside a quote, do not add ellipses, and \
do not join text from two places.
4. If the passages do not contain the answer, set `sufficient` to false. This is the correct, \
expected outcome for an unanswerable question, not a failure.

Reply with one JSON object and nothing else:

{"sufficient": true, "answer": "<answer with inline [n] citations>", \
"support": [{"passage": <n>, "quote": "<verbatim span from passage n>"}]}

or:

{"sufficient": false, "answer": "", "support": []}\
"""

USER_TEMPLATE = """\
Question: {question}

Passages:
{passages}\
"""

PASSAGE_TEMPLATE = "[{number}] (source: {label})\n{text}"


def render_user(question: str, passages: list[tuple[int, str, str]]) -> str:
    """Build the user message.

    `passages` is `(number, label, text)`. The label is shown so the model can refer to a
    source naturally, but citations are keyed on the number — a label is not unique and a
    model that cites by file name cannot be checked against a specific chunk.
    """
    rendered = "\n\n".join(
        PASSAGE_TEMPLATE.format(number=n, label=label, text=text) for n, label, text in passages
    )
    return USER_TEMPLATE.format(question=question, passages=rendered)
