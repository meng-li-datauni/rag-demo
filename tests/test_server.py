import pytest
from fastapi.testclient import TestClient

import monitoring
import server
from conftest import FakeChatModel


@pytest.fixture
def client(rag, fake_llms, tmp_path, monkeypatch):
    monkeypatch.setattr(monitoring, "DB_PATH", tmp_path / "monitoring.db")
    monkeypatch.setattr(server, "RagPipeline", lambda: rag)
    monkeypatch.setattr(rag, "llm", FakeChatModel())
    with TestClient(server.app) as c:
        yield c


@pytest.mark.parametrize("mode,llm_calls", [("rag", 1), ("agent", 2), ("multi", 4)])
def test_chat_each_mode_reports_usage(client, mode, llm_calls):
    r = client.post("/api/chat", json={"question": "How many PTO days?", "mode": mode})
    assert r.status_code == 200
    data = r.json()
    assert "27 days" in data["answer"]
    assert data["usage"]["llm_calls"] == llm_calls
    assert data["usage"]["cost_usd"] > 0
    assert (data["thread_id"] is None) == (mode == "rag")


def test_empty_question_rejected(client):
    assert client.post("/api/chat", json={"question": "  "}).status_code == 400


def test_approval_flow(client):
    r = client.post("/api/chat", json={"question": "Open a ticket please", "mode": "agent"}).json()
    assert r["pending_approval"][0]["tool"] == "create_support_ticket"
    done = client.post("/api/resume", json={"mode": "agent", "thread_id": r["thread_id"],
                                            "decision": "approve"})
    assert done.status_code == 200 and done.json()["pending_approval"] is None
    again = client.post("/api/resume", json={"mode": "agent", "thread_id": r["thread_id"],
                                             "decision": "approve"})
    assert again.status_code == 409


def test_feedback_and_monitoring(client):
    request_id = client.post("/api/chat", json={"question": "PTO?"}).json()["request_id"]
    assert client.post("/api/feedback", json={"request_id": request_id, "score": 1}).status_code == 200
    assert client.post("/api/feedback", json={"request_id": "nope", "score": 1}).status_code == 404
    totals = client.get("/api/monitoring").json()["totals"]
    assert totals["requests"] == 1 and totals["thumbs_up"] == 1


def test_eval_estimate_and_run(client):
    assert client.get("/api/eval/estimate?mode=rag&limit=2").json()["based_on"] == "a rough default"
    run = client.post("/api/eval/run", json={"mode": "rag", "limit": 2}).json()
    assert len(run["results"]) == 2
    assert run["results"][0]["answer_ok"]  # fake answer "27 days" matches Q1
    assert run["retrieval_accuracy"] == 1.0
    estimate = client.get("/api/eval/estimate?mode=rag&limit=2").json()
    assert estimate["based_on"] == "your past requests"
