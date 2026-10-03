import pytest

import agents
from agents import AGENT_MAX_MODEL_CALLS, build_agent, build_multi_agent
from conftest import FakeChatModel


def test_agent_searches_then_answers(rag, fake_llms):
    result = build_agent(rag.vector_store).ask("How many PTO days?", "t1")
    assert "27 days" in result.answer
    assert [s["tool"] for s in result.steps] == ["search_documents"]
    assert result.sources and result.pending_approval is None


@pytest.mark.parametrize("decision,created", [("approve", True), ("reject", False)])
def test_ticket_waits_for_human_approval(rag, fake_llms, decision, created):
    runner = build_agent(rag.vector_store)
    before = len(agents.TICKETS)

    paused = runner.ask("Please open a ticket: my stipend receipt was rejected", "t-hitl")
    assert paused.answer == ""
    assert paused.pending_approval[0]["tool"] == "create_support_ticket"
    assert len(agents.TICKETS) == before  # nothing happens until a human decides

    runner.resume("t-hitl", decision)
    assert len(agents.TICKETS) == before + (1 if created else 0)


def test_resume_without_pending_approval_fails(rag, fake_llms):
    with pytest.raises(ValueError):
        build_agent(rag.vector_store).resume("no-such-thread", "approve")


def test_agent_remembers_thread_and_redacts_email(rag, fake_llms):
    runner = build_agent(rag.vector_store)
    runner.ask("My email is anna@example.com. How many PTO days?", "t-mem")
    runner.ask("And the equipment stipend?", "t-mem")
    history = runner.graph.get_state({"configurable": {"thread_id": "t-mem"}}).values["messages"]
    human = [m.content for m in history if m.type == "human"]
    assert len(human) == 2  # second question sees the first
    assert "anna@example.com" not in human[0] and "[REDACTED_EMAIL]" in human[0]


def test_agent_stops_at_model_call_limit(rag, fake_llms):
    fake_llms["loop"] = True  # a model that never stops calling tools
    result = build_agent(rag.vector_store).ask("How many PTO days?", "t-loop")
    # Each allowed model call made one tool call; the next model call was blocked.
    assert len(result.steps) == AGENT_MAX_MODEL_CALLS
    assert "limit" in result.answer.lower()


def test_multi_agent_delegates_to_scoped_subagent(rag, fake_llms):
    fake_llms["delegate_to"] = "sales-agent"
    result = build_multi_agent(rag.vector_store).ask("What was Q3 revenue?", "m1")
    assert [(s["agent"], s["tool"]) for s in result.steps] == [
        ("supervisor", "task"), ("sales-agent", "search_sales_documents"),
    ]
    assert {s["source"] for s in result.sources} == {"q3_2026_sales_report.pdf"}


def test_multi_agent_tools_are_trimmed(rag, monkeypatch):
    captured: dict[str, set] = {}

    class Capture(FakeChatModel):
        def bind_tools(self, tools, **kw):
            names = [getattr(t, "name", None) or t.get("name") for t in tools]
            captured.setdefault(self.model_name, set()).update(names)
            return super().bind_tools(tools, **kw)

    monkeypatch.setattr(agents, "make_llm", lambda model, **kw: Capture(model_name=model))
    build_multi_agent(rag.vector_store).ask("How many PTO days?", "m2")
    supervisor_tools = captured["claude-sonnet-5-5"]
    assert {"task", "read_file", "create_support_ticket"} <= supervisor_tools
    assert not supervisor_tools & {"edit_file", "delete", "glob", "grep", "execute"}
    # Tools are bound when an agent runs; only the HR sub-agent ran here.
    assert captured["claude-haiku-4-5"] == {"search_hr_documents"}
