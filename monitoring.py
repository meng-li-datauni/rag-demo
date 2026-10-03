"""Local monitoring: token usage, cost and latency per request, plus user feedback.

Every Claude call reports its token counts; `UsageTracker` (a LangChain callback)
collects them for one request - including agent steps, sub-agents and summarization -
and `MonitoringStore` keeps one row per request in a local SQLite file.
"""

import json
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

DB_PATH = Path(__file__).parent / "monitoring.db"

# USD per million tokens (input, output). Cache reads cost 0.1x input, cache writes 1.25x.
PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


def langsmith_enabled() -> bool:
    return (os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
            and bool(os.environ.get("LANGSMITH_API_KEY")))


@dataclass
class Usage:
    llm_calls: int = 0
    input_tokens: int = 0       # total, including cached tokens
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    models: set[str] = field(default_factory=set)


def call_cost(model: str, input_tokens: int, output_tokens: int,
              cache_read: int, cache_write: int) -> float:
    price_in, price_out = next(
        (p for name, p in PRICES.items() if model.startswith(name)), (0.0, 0.0)
    )
    uncached = input_tokens - cache_read - cache_write
    return (uncached * price_in + cache_read * price_in * 0.1
            + cache_write * price_in * 1.25 + output_tokens * price_out) / 1_000_000


class UsageTracker(BaseCallbackHandler):
    """Sums token usage over every LLM call made while handling one request."""

    def __init__(self):
        self.usage = Usage()
        self._lock = threading.Lock()

    def on_llm_end(self, response, **kwargs) -> None:
        for generations in response.generations:
            for gen in generations:
                message = getattr(gen, "message", None)
                usage = getattr(message, "usage_metadata", None)
                if not usage:
                    continue
                meta = message.response_metadata
                model = meta.get("model_name") or meta.get("model") or "unknown"
                details = usage.get("input_token_details") or {}
                cache_read = details.get("cache_read") or 0
                cache_write = details.get("cache_creation") or 0
                with self._lock:
                    u = self.usage
                    u.llm_calls += 1
                    u.input_tokens += usage["input_tokens"]
                    u.output_tokens += usage["output_tokens"]
                    u.cache_read_tokens += cache_read
                    u.cache_write_tokens += cache_write
                    u.cost_usd += call_cost(model, usage["input_tokens"],
                                            usage["output_tokens"], cache_read, cache_write)
                    u.models.add(model)


class MonitoringStore:
    def __init__(self, path: Path = DB_PATH):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS requests (
                    id TEXT PRIMARY KEY, ts REAL, source TEXT, mode TEXT, thread_id TEXT,
                    question TEXT, answer TEXT, error TEXT, models TEXT,
                    llm_calls INTEGER, input_tokens INTEGER, output_tokens INTEGER,
                    cache_read_tokens INTEGER, cache_write_tokens INTEGER,
                    cost_usd REAL, latency_ms INTEGER, feedback INTEGER
                )""")
            self._conn.commit()

    def record(self, *, request_id: str, source: str, mode: str, thread_id: str | None,
               question: str, answer: str, error: str | None, usage: Usage,
               latency_ms: int) -> None:
        row = asdict(usage)
        row["models"] = json.dumps(sorted(usage.models))
        with self._lock:
            self._conn.execute(
                """INSERT INTO requests VALUES (:id, :ts, :source, :mode, :thread_id,
                   :question, :answer, :error, :models, :llm_calls, :input_tokens,
                   :output_tokens, :cache_read_tokens, :cache_write_tokens, :cost_usd,
                   :latency_ms, NULL)""",
                {**row, "id": request_id, "ts": time.time(), "source": source, "mode": mode,
                 "thread_id": thread_id, "question": question, "answer": answer,
                 "error": error, "latency_ms": latency_ms},
            )
            self._conn.commit()

    def set_feedback(self, request_id: str, score: int) -> bool:
        with self._lock:
            cur = self._conn.execute("UPDATE requests SET feedback = ? WHERE id = ?",
                                     (score, request_id))
            self._conn.commit()
            return cur.rowcount > 0

    def summary(self, recent: int = 50) -> dict:
        with self._lock:
            by_mode = [dict(r) for r in self._conn.execute("""
                SELECT mode, COUNT(*) AS requests, SUM(llm_calls) AS llm_calls,
                       SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens,
                       SUM(cost_usd) AS cost_usd, AVG(cost_usd) AS avg_cost_usd,
                       AVG(latency_ms) AS avg_latency_ms,
                       SUM(error IS NOT NULL) AS errors,
                       SUM(feedback = 1) AS thumbs_up, SUM(feedback = -1) AS thumbs_down
                FROM requests GROUP BY mode ORDER BY mode""")]
            rows = [dict(r) for r in self._conn.execute(
                "SELECT * FROM requests ORDER BY ts DESC LIMIT ?", (recent,))]
        for r in rows:
            r["models"] = json.loads(r["models"] or "[]")
        totals = {k: sum((m[k] or 0) for m in by_mode)
                  for k in ("requests", "llm_calls", "input_tokens", "output_tokens",
                            "cost_usd", "errors", "thumbs_up", "thumbs_down")}
        return {"totals": totals, "by_mode": by_mode, "recent": rows}

    def avg_cost(self, mode: str) -> float | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT AVG(cost_usd) FROM requests WHERE mode = ? AND error IS NULL "
                "AND llm_calls > 0", (mode,)).fetchone()
        return row[0]
