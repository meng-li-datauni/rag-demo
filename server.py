"""FastAPI backend + chat UI for the RAG demo.

Run with: .venv/bin/uvicorn server:app --reload
Then open http://localhost:8000
"""

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agents import AGENT_MODEL, SUBAGENT_MODEL, AgentResult, build_agent, build_multi_agent
from evaluate import EVAL_CASES, score_case
from monitoring import MonitoringStore, UsageTracker, langsmith_enabled
from rag import DOCS_DIR, LLM_MODEL, RagPipeline

STATIC_DIR = Path(__file__).parent / "static"

Mode = Literal["rag", "agent", "multi"]

# Rough cost per question used until the monitoring data has real averages.
DEFAULT_COST_PER_QUESTION = {"rag": 0.001, "agent": 0.01, "multi": 0.03}


class ChatRequest(BaseModel):
    question: str
    mode: Mode = "rag"
    thread_id: str | None = None  # agent modes: same id = same conversation memory


class ResumeRequest(BaseModel):
    mode: Literal["agent", "multi"]
    thread_id: str
    decision: Literal["approve", "reject"]


class FeedbackRequest(BaseModel):
    request_id: str
    score: Literal[1, -1]


class EvalRequest(BaseModel):
    mode: Mode = "rag"
    limit: int = Field(default=3, ge=1, le=len(EVAL_CASES))


class Source(BaseModel):
    source: str
    page: int
    text: str


class Step(BaseModel):
    agent: str
    tool: str
    args: dict


class PendingAction(BaseModel):
    tool: str
    args: dict


class UsageInfo(BaseModel):
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    steps: list[Step] = []
    pending_approval: list[PendingAction] | None = None
    thread_id: str | None = None
    request_id: str
    usage: UsageInfo


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the index once at startup; the agents share its vector store.
    rag = RagPipeline()
    app.state.rag = rag
    app.state.agents = {
        "agent": build_agent(rag.vector_store),
        "multi": build_multi_agent(rag.vector_store),
    }
    app.state.monitoring = MonitoringStore()
    yield


app = FastAPI(title="RAG Demo", lifespan=lifespan)


def run_tracked(mode: Mode, question: str, thread_id: str | None, *, source: str = "chat",
                resume_decision: str | None = None) -> ChatResponse:
    """Run one request with usage tracking, record it, and return the response.

    The request id doubles as the LangSmith run id, so UI feedback can be attached
    to the matching trace.
    """
    request_id = str(uuid.uuid4())
    tracker = UsageTracker()
    config = {
        "callbacks": [tracker],
        "run_id": request_id,
        "run_name": f"{mode}-{source}",
        "tags": [mode, source],
        # LangSmith groups traces with the same thread_id into one conversation.
        "metadata": {"mode": mode, "source": source,
                     **({"thread_id": thread_id} if thread_id else {})},
    }
    start = time.monotonic()
    answer, error = "", None
    try:
        if mode == "rag":
            answer, retrieved = app.state.rag.ask(question, config)
            result = AgentResult(answer=answer, sources=[
                {"source": d.metadata["source"], "page": d.metadata["page"],
                 "text": d.page_content} for d in retrieved
            ])
        else:
            runner = app.state.agents[mode]
            if resume_decision:
                result = runner.resume(thread_id, resume_decision, config)
            else:
                result = runner.ask(question, thread_id, config)
        answer = result.answer or ("(waiting for approval)" if result.pending_approval else "")
        return ChatResponse(
            answer=result.answer, sources=result.sources, steps=result.steps,
            pending_approval=result.pending_approval, thread_id=thread_id,
            request_id=request_id,
            usage=UsageInfo(llm_calls=tracker.usage.llm_calls,
                            input_tokens=tracker.usage.input_tokens,
                            output_tokens=tracker.usage.output_tokens,
                            cost_usd=round(tracker.usage.cost_usd, 6),
                            latency_ms=int((time.monotonic() - start) * 1000)),
        )
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        app.state.monitoring.record(
            request_id=request_id, source=source, mode=mode, thread_id=thread_id,
            question=question, answer=answer, error=error, usage=tracker.usage,
            latency_ms=int((time.monotonic() - start) * 1000),
        )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/documents")
def documents() -> dict:
    return {
        "documents": [p.name for p in sorted(DOCS_DIR.glob("*.pdf"))],
        "num_chunks": app.state.rag.num_chunks,
        "models": {"rag": LLM_MODEL, "agent": AGENT_MODEL,
                   "multi": f"{AGENT_MODEL} + {SUBAGENT_MODEL} sub-agents"},
        "langsmith": langsmith_enabled(),
    }


# Plain `def` endpoints so FastAPI runs the blocking LLM calls in a worker thread.
@app.post("/api/chat")
def chat(req: ChatRequest) -> ChatResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty")
    thread_id = None if req.mode == "rag" else (req.thread_id or str(uuid.uuid4()))
    try:
        return run_tracked(req.mode, question, thread_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}") from e


@app.post("/api/resume")
def resume(req: ResumeRequest) -> ChatResponse:
    try:
        return run_tracked(req.mode, f"[{req.decision}]", req.thread_id,
                           resume_decision=req.decision)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}") from e


@app.post("/api/feedback")
def feedback(req: FeedbackRequest) -> dict:
    if not app.state.monitoring.set_feedback(req.request_id, req.score):
        raise HTTPException(status_code=404, detail="Unknown request id")
    sent_to_langsmith = False
    if langsmith_enabled():
        try:
            from langsmith import Client
            Client().create_feedback(req.request_id, key="user_score",
                                     score=1 if req.score == 1 else 0)
            sent_to_langsmith = True
        except Exception:
            pass  # local feedback is already saved; LangSmith is best-effort
    return {"ok": True, "langsmith": sent_to_langsmith}


@app.get("/api/monitoring")
def monitoring() -> dict:
    return app.state.monitoring.summary()


@app.get("/api/eval/estimate")
def eval_estimate(mode: Mode = "rag", limit: int = 3) -> dict:
    limit = max(1, min(limit, len(EVAL_CASES)))
    measured = app.state.monitoring.avg_cost(mode)
    per_question = measured if measured is not None else DEFAULT_COST_PER_QUESTION[mode]
    return {"mode": mode, "limit": limit, "total_questions": len(EVAL_CASES),
            "cost_per_question": per_question, "estimated_cost": per_question * limit,
            "based_on": "your past requests" if measured is not None else "a rough default"}


@app.post("/api/eval/run")
def eval_run(req: EvalRequest) -> dict:
    results = []
    for question, expected_sources, facts in EVAL_CASES[:req.limit]:
        thread_id = None if req.mode == "rag" else str(uuid.uuid4())
        try:
            res = run_tracked(req.mode, question, thread_id, source="eval")
            retrieved = {s.source for s in res.sources}
            score = score_case(expected_sources, facts, res.answer, retrieved)
            results.append({"question": question, "answer": res.answer,
                            "expected_sources": expected_sources,
                            "retrieved_sources": sorted(retrieved), **score,
                            "cost_usd": res.usage.cost_usd,
                            "latency_ms": res.usage.latency_ms})
        except Exception as e:
            results.append({"question": question, "answer": f"Error: {e}",
                            "expected_sources": expected_sources, "retrieved_sources": [],
                            "retrieval_ok": False, "answer_ok": False, "missing": facts,
                            "cost_usd": 0, "latency_ms": 0})
    return {
        "mode": req.mode,
        "results": results,
        "retrieval_accuracy": sum(r["retrieval_ok"] for r in results) / len(results),
        "answer_accuracy": sum(r["answer_ok"] for r in results) / len(results),
        "total_cost_usd": sum(r["cost_usd"] for r in results),
    }
