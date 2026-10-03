"""Shared test setup. The whole suite runs offline: no Claude calls, no LangSmith traces."""

import os

# Must happen before project modules load .env (load_dotenv never overrides set vars).
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["ANTHROPIC_API_KEY"] = "test-key-not-real"  # an accidental real call fails, not spends

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import agents
from rag import RagPipeline

USAGE = {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100,
         "input_token_details": {"cache_read": 400}}


class FakeChatModel(BaseChatModel):
    """Scripted stand-in for Claude.

    With tools bound and no tool result yet, it calls a tool: `create_support_ticket`
    if the question mentions a ticket, `task` (delegating to `delegate_to`) if
    available, otherwise the first `search_*` tool. After a tool result it answers
    with `answer`. With `loop=True` it never stops calling tools.
    """

    model_name: str = "claude-haiku-4-5"
    answer: str = "Employees get 27 days. [employee_handbook.pdf]"
    delegate_to: str = "hr-agent"
    loop: bool = False
    tool_names: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _get_ls_params(self, **kwargs):
        return {"ls_provider": "anthropic", "ls_model_name": self.model_name}

    def bind_tools(self, tools, **kwargs):
        names = [getattr(t, "name", None) or t.get("name") for t in tools]
        return self.model_copy(update={"tool_names": names})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        question = next(m for m in reversed(messages) if m.type == "human").content
        wants_tool = self.tool_names and (self.loop or messages[-1].type != "tool")
        if wants_tool:
            if "ticket" in question.lower() and "create_support_ticket" in self.tool_names:
                call = {"name": "create_support_ticket",
                        "args": {"title": "Stipend", "description": question, "team": "hr"}}
            elif "task" in self.tool_names:
                call = {"name": "task",
                        "args": {"subagent_type": self.delegate_to, "description": question}}
            else:
                name = next(n for n in self.tool_names if n.startswith("search"))
                call = {"name": name, "args": {"query": question}}
            message = AIMessage(content="", tool_calls=[{**call, "id": f"call_{len(messages)}"}])
        else:
            message = AIMessage(content=self.answer)
        message.usage_metadata = USAGE
        message.response_metadata = {"model_name": self.model_name}
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.fixture(scope="session")
def rag() -> RagPipeline:
    """Real index over docs/ (local embeddings), built once for the whole run."""
    return RagPipeline()


@pytest.fixture
def fake_llms(monkeypatch):
    """Make every agent model a FakeChatModel; returns a dict of options to tweak."""
    options: dict = {}
    monkeypatch.setattr(
        agents, "make_llm",
        lambda model, **kw: FakeChatModel(model_name=model, **options),
    )
    return options
