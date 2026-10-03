"""Agent and multi-agent versions of the RAG demo.

* Agent (`create_agent`): one Claude agent with a document-search tool, a ticket tool
  that needs human approval, per-thread memory, and middleware.
* Multi-agent (`deepagents`): a supervisor that delegates to HR, Engineering and Sales
  sub-agents, each limited to its own PDFs, with AGENTS.md memory and a SKILL.md skill.
"""

import itertools
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from deepagents import (
    FilesystemPermission,
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.backends import FilesystemBackend
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
)
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from rag import NO_ANSWER, TOP_K

AGENT_MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")
SUBAGENT_MODEL = os.environ.get("SUBAGENT_MODEL", "claude-haiku-4-5")
SUMMARY_MODEL = "claude-haiku-4-5"
WORKSPACE_DIR = Path(__file__).parent / "agent_workspace"

TICKET_APPROVAL = {"create_support_ticket": {"allowed_decisions": ["approve", "reject"]}}


def make_llm(model: str, effort: str = "medium") -> ChatAnthropic:
    if model.startswith("claude-haiku"):
        return ChatAnthropic(model=model, max_tokens=8192)
    # Summarization rewrites earlier turns, which would invalidate thinking blocks on
    # newer models; "drop_block" tells the API to drop those blocks instead of erroring.
    return ChatAnthropic(
        model=model,
        max_tokens=16000,
        betas=["thinking-binding-controls-2026-08-01"],
        thinking={"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}},
        output_config={"effort": effort},
    )


# --- Run log: tools and middleware record what happened during one request ---------

_run_log: ContextVar[dict | None] = ContextVar("run_log", default=None)


def _log(key: str, item: dict) -> None:
    log = _run_log.get()
    if log is not None:
        log[key].append(item)


class StepLogger(AgentMiddleware):
    """Custom middleware: records every tool call so the UI can show the agent's steps."""

    def __init__(self, agent_label: str):
        super().__init__()
        self.agent_label = agent_label

    @property
    def name(self) -> str:
        return f"StepLogger[{self.agent_label}]"

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        _log("steps", {"agent": self.agent_label, "tool": call["name"], "args": call["args"]})
        return handler(request)


# --- Tools ---------------------------------------------------------------------------

def make_search_tool(vector_store: InMemoryVectorStore, name: str, description: str,
                     sources: set[str] | None = None):
    """Document search over the vector store, optionally limited to some PDFs."""
    doc_filter = (lambda doc: doc.metadata["source"] in sources) if sources else None

    @tool(name, description=description)
    def search(query: str) -> str:
        docs = vector_store.similarity_search(query, k=TOP_K, filter=doc_filter)
        for d in docs:
            _log("sources", {"source": d.metadata["source"], "page": d.metadata["page"],
                             "text": d.page_content})
        if not docs:
            return "No matching documents."
        return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)

    return search


_ticket_ids = itertools.count(1001)
TICKETS: list[dict] = []


@tool
def create_support_ticket(title: str, description: str, team: Literal["hr", "engineering", "sales"]) -> str:
    """Create an internal support ticket. Only use when the user explicitly asks for a ticket."""
    ticket = {"id": f"TCK-{next(_ticket_ids)}", "title": title, "description": description,
              "team": team}
    TICKETS.append(ticket)
    return f"Created ticket {ticket['id']} for the {team} team."


# --- Running an agent ----------------------------------------------------------------

@dataclass
class AgentResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    pending_approval: list[dict] | None = None  # tool calls waiting for a human decision


class AgentRunner:
    def __init__(self, graph):
        self.graph = graph

    def ask(self, question: str, thread_id: str, config: dict | None = None) -> AgentResult:
        return self._run({"messages": [{"role": "user", "content": question}]}, thread_id,
                         config)

    def resume(self, thread_id: str, decision: Literal["approve", "reject"],
               config: dict | None = None) -> AgentResult:
        thread_config = {"configurable": {"thread_id": thread_id}}
        interrupts = self.graph.get_state(thread_config).interrupts
        if not interrupts:
            raise ValueError("Nothing is waiting for approval in this thread")
        pending = interrupts[0].value["action_requests"]
        decisions = [{"type": decision} for _ in pending]
        return self._run(Command(resume={"decisions": decisions}), thread_id, config)

    def _run(self, graph_input, thread_id: str, config: dict | None = None) -> AgentResult:
        # Extra config: callbacks (usage tracking), run_id, tags and metadata (LangSmith).
        config = {**(config or {}), "configurable": {"thread_id": thread_id}}
        log = {"sources": [], "steps": []}
        token = _run_log.set(log)
        try:
            state = self.graph.invoke(graph_input, config)
        finally:
            _run_log.reset(token)

        result = AgentResult(answer="", sources=_dedupe(log["sources"]), steps=log["steps"])
        if interrupts := state.get("__interrupt__"):
            result.pending_approval = [
                {"tool": a["name"], "args": a["args"]}
                for a in interrupts[0].value["action_requests"]
            ]
        else:
            result.answer = state["messages"][-1].text
        return result


def _dedupe(sources: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for s in sources:
        key = (s["source"], s["text"])
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


# --- Agent (create_agent) --------------------------------------------------------------

AGENT_PROMPT = (
    "You are the internal assistant of Halvorsen Robotics. Answer questions using ONLY "
    "results from the search_documents tool; never use outside knowledge. Search as many "
    "times as you need, e.g. once per sub-question. If the documents don't contain the "
    f"answer, reply exactly: \"{NO_ANSWER}\" Keep answers short and end with the source "
    "file name(s) in square brackets. Create a support ticket only when the user "
    "explicitly asks for one."
)


def build_agent(vector_store: InMemoryVectorStore) -> AgentRunner:
    search = make_search_tool(
        vector_store, "search_documents",
        "Search Halvorsen Robotics' internal documents (handbook, Project Kestrel spec, "
        "sales report, incident postmortem). Returns matching excerpts with file names.",
    )
    graph = create_agent(
        make_llm(AGENT_MODEL),
        tools=[search, create_support_ticket],
        system_prompt=AGENT_PROMPT,
        middleware=[
            # Redact email addresses the user types before they reach the model.
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            # Keep long chats within budget: summarize older messages.
            SummarizationMiddleware(make_llm(SUMMARY_MODEL), trigger=("tokens", 6000),
                                    keep=("messages", 12)),
            # Pause for a human before any ticket is created.
            HumanInTheLoopMiddleware(interrupt_on=TICKET_APPROVAL),
            StepLogger("agent"),
            # Cache the system prompt + tool definitions between calls (cheaper re-reads).
            AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        ],
        checkpointer=InMemorySaver(),  # short-term memory per thread_id
        name="agent",
    )
    return AgentRunner(graph)


# --- Multi-agent (deepagents) --------------------------------------------------------

SUBAGENTS = [
    ("hr-agent", "HR specialist: employee handbook - paid time off, hybrid work, "
     "equipment stipend, on-call pay.", {"employee_handbook.pdf"}),
    ("engineering-agent", "Engineering specialist: Project Kestrel technical spec and "
     "incident postmortems (e.g. INC-0917).",
     {"project_kestrel_spec.pdf", "incident_postmortem_inc0917.pdf"}),
    ("sales-agent", "Sales specialist: quarterly sales reports - revenue, regions, "
     "customers, outlook.", {"q3_2026_sales_report.pdf"}),
]

SUBAGENT_PROMPT = (
    "You are the {name} at Halvorsen Robotics. Answer the task using ONLY results from "
    "your search tool. Report the facts you found together with their source file names. "
    "If your documents don't contain the answer, say so plainly."
)

SUPERVISOR_PROMPT = (
    "You are the supervisor of Halvorsen Robotics' internal assistant. Delegate every "
    "factual question to the right specialist sub-agent(s) with the task tool; don't "
    "answer from your own knowledge. Follow the rules in AGENTS.md and use a skill when "
    "one matches the request."
)


# Drop deep-agent tools this demo never uses: every tool definition is re-sent on every
# call, so fewer tools = fewer input tokens. read_file stays (skills need it), as do
# ls/write_file (workspace + notes). The default general-purpose sub-agent is unused too.
register_harness_profile("anthropic", HarnessProfile(
    excluded_tools=frozenset({"edit_file", "delete", "glob", "grep", "execute"}),
    general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
))


def build_multi_agent(vector_store: InMemoryVectorStore) -> AgentRunner:
    # Sub-agents are plain create_agent graphs with only their search tool - no file
    # tools or deep-agent prompt, so each sub-agent call stays small.
    subagents = []
    for name, description, sources in SUBAGENTS:
        team = name.split("-")[0]
        search = make_search_tool(
            vector_store, f"search_{team}_documents",
            f"Search the {team} documents ({', '.join(sorted(sources))}).", sources,
        )
        subagents.append({
            "name": name,
            "description": description,
            "runnable": create_agent(
                make_llm(SUBAGENT_MODEL),
                tools=[search],
                system_prompt=SUBAGENT_PROMPT.format(name=name),
                middleware=[StepLogger(name)],
                name=name,
            ),
        })

    graph = create_deep_agent(
        # Low effort: the supervisor mostly routes and combines, so deep thinking is wasted.
        model=make_llm(AGENT_MODEL, effort="low"),
        tools=[create_support_ticket],
        system_prompt=SUPERVISOR_PROMPT,
        middleware=[StepLogger("supervisor")],
        subagents=subagents,
        memory=["/AGENTS.md"],    # loaded into the system prompt
        skills=["/skills/"],      # loaded on demand (progressive disclosure)
        # File tools are confined to agent_workspace/ ...
        backend=FilesystemBackend(root_dir=WORKSPACE_DIR, virtual_mode=True),
        # ... and may only write under /notes/ (first matching rule wins).
        permissions=[
            FilesystemPermission(operations=["write"], paths=["/notes/**"], mode="allow"),
            FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
        ],
        interrupt_on=TICKET_APPROVAL,
        checkpointer=InMemorySaver(),
        name="multi-agent",
    )
    return AgentRunner(graph)
