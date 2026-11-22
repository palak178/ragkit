"""Command line.

    ragkit ingest handbook.pdf notes.md
    ragkit ask "How much notice is required?"
    ragkit serve

`ingest` and `ask` are separate processes, so with the default in-memory store the index does
not survive between them — `ask` would find nothing. Rather than pretend otherwise, `ask`
takes the documents to index first (`--doc`), and the README points at `RAGKIT_STORE=qdrant`
for a persistent index. Silently returning "no answer" because the store was empty is the
kind of confusing-by-default behaviour this project is a reaction to.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer

from ragkit.errors import RagkitError
from ragkit.pipeline import RagPipeline

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Grounded RAG: cited answers, or an explicit refusal.",
)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,  # stderr, so `--json` output stays machine-readable on stdout.
    )


@app.command()
def ingest(
    paths: list[Path] = typer.Argument(..., help="Documents to index (.pdf, .txt, .md)."),
    as_json: bool = typer.Option(False, "--json", help="Emit the reports as JSON."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Index one or more documents."""
    _configure_logging(verbose)
    pipeline = RagPipeline.from_settings()
    reports = []
    failures = 0
    for path in paths:
        try:
            reports.append(pipeline.ingest(path))
        except RagkitError as exc:
            failures += 1
            typer.secho(f"{path}: {exc}", fg=typer.colors.RED, err=True)

    if as_json:
        typer.echo(json.dumps([r.model_dump() for r in reports], indent=2))
    else:
        for r in reports:
            typer.echo(f"{r.doc_id}: {r.chunks_written} chunks from {r.pages} page(s)")
        if pipeline.store.count():
            typer.echo(f"index now holds {pipeline.store.count()} chunk(s)")

    # Non-zero exit when any document failed, so a shell loop or CI step notices.
    if failures:
        raise typer.Exit(code=1)


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to answer."),
    docs: list[Path] = typer.Option(
        [],
        "--doc",
        "-d",
        help="Index these documents first. Required with the default in-memory store, "
        "since nothing persists between commands.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the full answer as JSON."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Answer a question from the index."""
    _configure_logging(verbose)
    pipeline = RagPipeline.from_settings()
    for path in docs:
        try:
            pipeline.ingest(path)
        except RagkitError as exc:
            typer.secho(f"{path}: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

    if pipeline.store.count() == 0:
        typer.secho(
            "The index is empty. Pass --doc FILE, or configure a persistent store.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=2)

    answer = pipeline.ask(question)

    if as_json:
        typer.echo(answer.model_dump_json(indent=2))
        raise typer.Exit(code=0 if not answer.refused else 3)

    if answer.refused:
        typer.secho(answer.refusal_reason or "No grounded answer.", fg=typer.colors.YELLOW)
        typer.echo(
            f"  verdict={answer.groundedness.verdict} "
            f"best_score={answer.groundedness.best_score:.3f} "
            f"threshold={answer.groundedness.score_threshold:.3f}"
        )
        # Exit 3 distinguishes "worked, found nothing" from a crash (1) and misuse (2).
        raise typer.Exit(code=3)

    typer.echo(answer.rendered())
    typer.echo()
    for citation in answer.citations:
        typer.secho(f"  [{citation.marker}] “{citation.quote[:100]}”", fg=typer.colors.BRIGHT_BLACK)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address. Localhost by default."),
    port: int = typer.Option(8000),
) -> None:
    """Run the HTTP API."""
    try:
        import uvicorn
    except ImportError as exc:
        typer.secho("Install the api extra: pip install '.[api]'", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    from ragkit.api import create_app

    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    app()
