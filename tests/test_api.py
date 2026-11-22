"""Tests for the HTTP surface.

The contract worth pinning down: a refusal is a **200 with a null answer**, because a question
the corpus cannot answer is a correct outcome rather than a fault. The path-traversal test is
here because `filename` on an upload is attacker-controlled.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.conftest import NOTICE_POLICY, ScriptedLLM, draft_json, make_pipeline, minimal_pdf

from ragkit.api import create_app


@pytest.fixture
def client(pipeline) -> TestClient:
    return TestClient(create_app(pipeline))


@pytest.fixture
def indexed_client(indexed_pipeline) -> TestClient:
    return TestClient(create_app(indexed_pipeline))


def upload(name: str, data: bytes) -> dict:
    return {"file": (name, io.BytesIO(data), "application/octet-stream")}


class TestHealth:
    def test_reports_the_stack(self, client: TestClient):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["chunks_indexed"] == 0
        assert body["embedder"] == "hashing-512"
        assert body["llm"] == "extractive-stub"

    def test_chunk_count_reflects_ingestion(self, indexed_client: TestClient):
        assert indexed_client.get("/health").json()["chunks_indexed"] > 0


class TestIngest:
    def test_uploads_markdown(self, client: TestClient):
        response = client.post("/ingest", files=upload("policy.md", NOTICE_POLICY.encode()))
        assert response.status_code == 201
        body = response.json()
        assert body["doc_id"] == "policy.md"
        assert body["chunks_written"] > 0

    def test_uploads_a_real_pdf(self, client: TestClient):
        pdf = minimal_pdf("Employees must give thirty days written notice.")
        response = client.post("/ingest", files=upload("handbook.pdf", pdf))
        assert response.status_code == 201
        assert response.json()["chunks_written"] == 1

    def test_unsupported_type_is_415(self, client: TestClient):
        response = client.post("/ingest", files=upload("book.epub", b"whatever"))
        assert response.status_code == 415
        assert ".pdf" in response.json()["detail"]

    def test_unreadable_document_is_422_not_500(self, client: TestClient):
        # Well-formed request, unusable document.
        response = client.post("/ingest", files=upload("blank.txt", b"   \n  "))
        assert response.status_code == 422
        assert "no text" in response.json()["detail"]

    def test_scanned_pdf_is_422_and_mentions_ocr(self, client: TestClient):
        response = client.post("/ingest", files=upload("scan.pdf", minimal_pdf(" ")))
        assert response.status_code == 422
        assert "OCR" in response.json()["detail"]

    def test_path_traversal_in_the_filename_is_neutralised(self, client: TestClient):
        # Only the final component may ever be used.
        response = client.post(
            "/ingest", files=upload("../../../../tmp/evil.md", NOTICE_POLICY.encode())
        )
        assert response.status_code == 201
        assert response.json()["doc_id"] == "evil.md"

    def test_absolute_path_in_the_filename_is_neutralised(self, client: TestClient):
        response = client.post("/ingest", files=upload("/etc/passwd.md", NOTICE_POLICY.encode()))
        assert response.status_code == 201
        assert response.json()["doc_id"] == "passwd.md"

    def test_ingest_then_ask_round_trip(self, client: TestClient):
        client.post("/ingest", files=upload("policy.md", NOTICE_POLICY.encode()))
        body = client.post(
            "/ask", json={"question": "How much notice must an employee give before resigning?"}
        ).json()
        assert body["text"] is not None
        assert "thirty days" in body["text"].lower()
        assert body["citations"][0]["doc_id"] == "policy.md"


class TestAsk:
    def test_grounded_answer_carries_citations_and_report(self, indexed_client: TestClient):
        response = indexed_client.post(
            "/ask", json={"question": "How many days of annual leave do employees accrue?"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["text"]
        assert body["citations"]
        assert body["groundedness"]["verdict"] == "grounded"
        assert body["citations"][0]["quote"]

    def test_refusal_is_200_with_a_null_answer(self, indexed_client: TestClient):
        response = indexed_client.post("/ask", json={"question": "What is the capital of Peru?"})
        assert response.status_code == 200, "a refusal is not an error"
        body = response.json()
        assert body["text"] is None
        assert body["citations"] == []
        assert body["groundedness"]["verdict"] == "insufficient_retrieval"
        assert body["refusal_reason"]

    def test_refusal_explains_itself_with_numbers(self, indexed_client: TestClient):
        body = indexed_client.post("/ask", json={"question": "Who invented the piano?"}).json()
        report = body["groundedness"]
        assert report["best_score"] < report["score_threshold"]

    def test_empty_question_is_rejected_by_validation(self, indexed_client: TestClient):
        assert indexed_client.post("/ask", json={"question": ""}).status_code == 422

    def test_missing_question_field(self, indexed_client: TestClient):
        assert indexed_client.post("/ask", json={}).status_code == 422

    def test_overlong_question_is_rejected(self, indexed_client: TestClient):
        assert indexed_client.post("/ask", json={"question": "x" * 2001}).status_code == 422

    def test_fabricated_answer_is_withheld_over_http(
        self, embedder, store, settings, policy_doc: Path
    ):
        llm = ScriptedLLM(
            draft_json(
                answer="Ninety days [1].",
                support=[{"passage": 1, "quote": "ninety days written notice is required"}],
            )
        )
        pipe = make_pipeline(embedder, store, settings, llm)
        pipe.ingest(policy_doc)
        client = TestClient(create_app(pipe))

        body = client.post("/ask", json={"question": "How much notice before resigning?"}).json()
        assert body["text"] is None
        assert body["groundedness"]["verdict"] == "ungrounded"


class TestErrorLeakage:
    def test_provider_failure_does_not_leak_internals(
        self, embedder, store, settings, policy_doc: Path
    ):
        class Exploding:
            model = "exploding"

            def complete(self, *, system: str, user: str) -> str:
                from ragkit.errors import GenerationError

                raise GenerationError("upstream request failed")

        pipe = make_pipeline(embedder, store, settings, Exploding())
        pipe.ingest(policy_doc)
        client = TestClient(create_app(pipe), raise_server_exceptions=False)

        response = client.post("/ask", json={"question": "How much notice is required?"})
        assert response.status_code == 502
        detail = response.json()["detail"]
        # The message is surfaced; a traceback, a base URL or a key must not be.
        assert "upstream request failed" in detail
        assert "Traceback" not in detail
        assert "api_key" not in detail.lower()
