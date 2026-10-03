import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from monitoring import MonitoringStore, Usage, UsageTracker, call_cost


def test_call_cost_uses_list_prices_and_cache_discount():
    # Haiku: 600 uncached * $1 + 400 cache reads * $0.10 + 100 output * $5, per million.
    assert call_cost("claude-haiku-4-5", 1000, 100, 400, 0) == pytest.approx(0.00114)
    # Cache writes cost 1.25x input.
    assert call_cost("claude-sonnet-5-5", 1000, 0, 0, 1000) == pytest.approx(0.0025)
    assert call_cost("some-unknown-model", 1000, 100, 0, 0) == 0


def test_usage_tracker_sums_calls():
    tracker = UsageTracker()
    message = AIMessage(content="hi", response_metadata={"model_name": "claude-haiku-4-5"},
                        usage_metadata={"input_tokens": 1000, "output_tokens": 100,
                                        "total_tokens": 1100,
                                        "input_token_details": {"cache_read": 400}})
    result = LLMResult(generations=[[ChatGeneration(message=message)]])
    tracker.on_llm_end(result)
    tracker.on_llm_end(result)
    u = tracker.usage
    assert (u.llm_calls, u.input_tokens, u.output_tokens, u.cache_read_tokens) == (2, 2000, 200, 800)
    assert u.cost_usd == pytest.approx(0.00228)
    assert u.models == {"claude-haiku-4-5"}


def test_store_records_feedback_and_summary(tmp_path):
    store = MonitoringStore(tmp_path / "test.db")
    usage = Usage(llm_calls=2, input_tokens=500, output_tokens=50, cost_usd=0.01,
                  models={"claude-haiku-4-5"})
    for i, mode in enumerate(["rag", "agent", "agent"]):
        store.record(request_id=f"r{i}", source="chat", mode=mode, thread_id=None,
                     question="q", answer="a", error=None, usage=usage, latency_ms=100)
    assert store.set_feedback("r1", 1)
    assert not store.set_feedback("missing", 1)

    summary = store.summary()
    assert summary["totals"]["requests"] == 3
    assert summary["totals"]["thumbs_up"] == 1
    assert {m["mode"]: m["requests"] for m in summary["by_mode"]} == {"agent": 2, "rag": 1}
    assert summary["recent"][0]["models"] == ["claude-haiku-4-5"]
    assert store.avg_cost("agent") == pytest.approx(0.01)
    assert store.avg_cost("multi") is None
