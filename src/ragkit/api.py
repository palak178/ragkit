"""HTTP surface. Needs the `api` extra.

Three endpoints, and the interesting one is `/ask`: a refusal is **200 with
`"answer": null`**, not an error status. A question the corpus cannot answer is a correct,
expected outcome of a working system, and modelling it as 4xx or 5xx would make callers treat
"I don't know" as a fault to retry around.

The response always carries the groundedness report, so a client can show why an answer was
withheld rather than just that it was.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ragkit import __version__
from ragkit.errors import ExtractionError, RagkitError, UnsupportedFormatError
from ragkit.extract import SUPPORTED_SUFFIXES
from ragkit.models import Answer, IngestReport
from ragkit.pipeline import RagPipeline

log = logging.getLogger("ragkit.api")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Health(BaseModel):
    status: str
    version: str
    chunks_indexed: int
    embedder: str
    llm: str


def create_app(pipeline: RagPipeline | None = None) -> FastAPI:
    """Build the app.

    `pipeline` is injectable so tests drive a fully in-memory stack — fake embedder, in-memory
    store, stub model — without monkey-patching module globals.
    """
    engine = pipeline or RagPipeline.from_settings()

    app = FastAPI(
        title="ragkit",
        version=__version__,
        description="Grounded RAG: cited answers, or an explicit refusal.",
    )

    @app.get("/health", response_model=Health)
    def health() -> Health:
        return Health(
            status="ok",
            version=__version__,
            chunks_indexed=engine.store.count(),
            embedder=engine.embedder.name,
            llm=engine.llm.model,
        )

    @app.post("/ingest", response_model=IngestReport, status_code=201)
    async def ingest(file: UploadFile = File(...)) -> IngestReport:
        # `filename` is attacker-controlled, so only its final component is ever used and the
        # file is written to a fresh temporary directory. Without this, a crafted
        # "../../etc/x" would be handed straight to a filesystem write.
        raw_name = Path(file.filename or "upload").name
        suffix = Path(raw_name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}",
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / raw_name
            target.write_bytes(await file.read())
            try:
                # doc_id is the uploaded name, so citations show the file the user
                # recognises rather than a temporary path.
                return engine.ingest(target, doc_id=raw_name)
            except UnsupportedFormatError as exc:
                raise HTTPException(status_code=415, detail=str(exc)) from exc
            except ExtractionError as exc:
                # 422: the request was well-formed, the document was not usable.
                raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/ask", response_model=Answer)
    def ask(request: AskRequest) -> Answer:
        try:
            return engine.ask(request.question)
        except RagkitError as exc:
            # Only this package's own errors are surfaced, and only their message. Anything
            # unexpected propagates to FastAPI's handler and is logged rather than echoed,
            # so a stack trace or a configured base URL never reaches a client.
            log.exception("ask failed")
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app
