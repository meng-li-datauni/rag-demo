# RAG Demo

A retrieval-augmented generation (RAG) demo built with LangChain and Claude. It has three modes (plain RAG, a single agent, and a multi-agent system), plus monitoring and evaluation both in the app and in LangSmith.

The knowledge base is a set of PDFs about **Halvorsen Robotics**, a fictional company. Every name, number and date in them is invented, so the model can't answer from its training data. A correct answer proves it came from the documents.

## How it works

```
PDFs ──► pypdf ──► text splitter ──► local embeddings ──► in-memory vector store
                                                                │
question ──► retriever (top 3 chunks) ──► prompt ──► Claude ──► answer + sources
```

| Step | Component |
|---|---|
| Load PDFs | `pypdf`, wrapped in LangChain `Document`s |
| Chunking | `RecursiveCharacterTextSplitter` (300 characters, 50 overlap) |
| Embeddings | `HuggingFaceEmbeddings` with `sentence-transformers/all-MiniLM-L6-v2`, run locally |
| Vector store | `InMemoryVectorStore` (rebuilt at startup) |
| LLM | `ChatAnthropic` with `claude-haiku-4-5` (override with the `LLM_MODEL` env var) |

The prompt tells Claude to answer only from the retrieved chunks and to cite the source file. If the chunks don't contain the answer, Claude must reply: *"I don't know based on the provided documents."*

### The three modes

| Mode | Built with | What happens per question | Models (env var) |
|---|---|---|---|
| **RAG** | `rag.py` | Search once, then one Claude call | Haiku 4.5 (`LLM_MODEL`) |
| **Agent** | `create_agent` | Claude decides when and how often to call `search_documents`, and loops until it can answer | Sonnet 5.5 (`AGENT_MODEL`) |
| **Multi-agent** | `deepagents` | A supervisor delegates to HR, Engineering and Sales sub-agents, each limited to its own PDFs, and combines their answers | Sonnet 5.5 supervisor (`AGENT_MODEL`), Haiku sub-agents (`SUBAGENT_MODEL`) |

The two agent modes share these features:

- **Human approval:** a `create_support_ticket` tool pauses and waits for Approve/Reject in the UI before it runs.
- **Memory:** each conversation (`thread_id`) keeps its history, using an in-memory checkpointer.
- **Middleware** (Agent mode): email addresses are redacted before they reach the model, long chats are summarized with Haiku, and a custom `StepLogger` records each tool call so the UI can show the agent's steps.
- **Deep-agent extras** (Multi-agent mode):
  - `agent_workspace/AGENTS.md` (instructions loaded into the system prompt)
  - `agent_workspace/skills/incident-summary/SKILL.md` (a skill loaded only when relevant)
  - file tools confined to `agent_workspace/`, with writes allowed only under `/notes/`
  - kept small to save tokens: the sub-agents are plain `create_agent` graphs with only their search tool, a harness profile removes the deep-agent tools this demo doesn't use (`edit_file`, `delete`, `glob`, `grep`, `execute`) along with the default general-purpose sub-agent, and the supervisor runs at `low` effort
- **Safety caps:** at most 6 Claude calls and 8 tool calls per question (4 calls per sub-agent). A looping agent stops with a "limit exceeded" message instead of spending more tokens. Change these in `agents.py` (`AGENT_MAX_MODEL_CALLS` and related constants).
- **Prompt caching:** the system prompt and tool definitions are cached between calls (`AnthropicPromptCachingMiddleware`; built into deepagents), so repeated reads within 5 minutes cost 10% of the normal input price

**Cost:** agent modes make several Claude calls per question: 2–3 for Agent, 4–6 for Multi-agent. The Monitoring tab shows the actual tokens and cost of each request. To keep costs down, set `AGENT_MODEL=claude-haiku-4-5` in `.env`.

## Project layout

| File | Purpose |
|---|---|
| `generate_docs.py` | Writes the fictional knowledge-base PDFs into `docs/` |
| `rag.py` | The RAG pipeline (`RagPipeline.ask()` returns the answer and the retrieved chunks) |
| `agents.py` | Agent (`create_agent`) and multi-agent (`deepagents`) versions |
| `agent_workspace/` | The deep agent's `AGENTS.md`, skills, and writable `notes/` folder |
| `monitoring.py` | Token/cost tracking callback and the local SQLite store (`monitoring.db`) |
| `evaluate.py` | Test questions, scoring, and the command-line evaluation |
| `langsmith_eval.py` | The same evaluation as a LangSmith dataset and experiment |
| `server.py` | FastAPI backend: chat, approval, feedback, monitoring and evaluation APIs |
| `static/index.html` | Web UI with Chat, Monitoring and Evaluation tabs (plain HTML/JS) |
| `main.py` | Terminal chat loop (RAG mode) |
| `tests/` | Offline test suite (`pytest`) |

### The documents

| File | Contents |
|---|---|
| `employee_handbook.pdf` | Paid time off, hybrid-work rules, equipment stipend, on-call pay |
| `project_kestrel_spec.pdf` | Specs, firmware and launch timeline of an internal drone project |
| `q3_2026_sales_report.pdf` | Revenue, regions, key customers, Q4 outlook |
| `incident_postmortem_inc0917.pdf` | Telemetry outage: root cause, fix, customer impact |

## Setup

Requires Python 3.10+ and an Anthropic API key **scoped to a workspace** (Console → Settings → API Keys → Create Key → pick a workspace).

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch, much smaller download
.venv/bin/pip install -r requirements.txt
.venv/bin/python generate_docs.py

cp .env.example .env   # then put your key in .env: ANTHROPIC_API_KEY=sk-ant-...
```

The embedding model (~90 MB) downloads from Hugging Face the first time it runs.

## Usage

### Web UI (FastAPI)

```bash
.venv/bin/uvicorn server:app --reload
```

Open http://127.0.0.1:8000. The app has three tabs:

- **Chat**: pick a mode in the sidebar. Each answer shows:
  - its sources and retrieved chunks
  - the agent's steps (in agent modes)
  - number of LLM calls, tokens, cost and response time
  - 👍/👎 buttons

  When an agent wants to create a ticket, Approve/Reject buttons appear. "New conversation" clears the agents' memory.
- **Monitoring**: totals and per-mode breakdown of requests, LLM calls, tokens, cost, response time, errors and 👍/👎 counts, plus the latest 50 requests. The data is stored locally in `monitoring.db` and survives restarts. Tracking costs no extra API calls, because the token counts come from Claude's own responses.
- **Evaluation**: pick a mode and how many test questions to run. The **Run** button shows the estimated cost first, based on your past requests in that mode. Results show pass/fail per question, plus answer accuracy, retrieval accuracy and the actual cost.

| Endpoint | Description |
|---|---|
| `GET /` | Web UI |
| `GET /api/documents` | Indexed PDFs, chunk count, models, whether LangSmith is on |
| `POST /api/chat` | `{"question", "mode": "rag"\|"agent"\|"multi", "thread_id"?}` → answer, sources, steps, pending approval, usage, `request_id` |
| `POST /api/resume` | `{"mode", "thread_id", "decision": "approve"\|"reject"}` → continues a paused agent |
| `POST /api/feedback` | `{"request_id", "score": 1\|-1}` → saved locally and, if enabled, sent to LangSmith |
| `GET /api/monitoring` | Totals, per-mode stats and recent requests |
| `GET /api/eval/estimate` | `?mode=&limit=` → estimated cost of an evaluation run |
| `POST /api/eval/run` | `{"mode", "limit"}` → per-question results and accuracy |

Interactive API docs: http://127.0.0.1:8000/docs

### Terminal

```bash
.venv/bin/python main.py                          # chat (RAG mode)
.venv/bin/python evaluate.py --mode rag --limit 3  # evaluation; drop --limit for all 13 questions
```

### Evaluation

The test set has 13 questions:

- **10 single-document questions**, each answered by exactly one PDF.
- **1 cross-document question**, which needs facts from the spec and the sales report.
- **2 unanswerable questions** (CEO salary, headcount). These pass only if the model says it doesn't know, which checks for hallucination.

Each question is scored on two things:

- **Retrieval:** did the retrieved chunks include the PDF that holds the answer?
- **Answer:** does the answer contain every expected fact? This is a case-insensitive keyword match, and `a|b` accepts either spelling.

The keyword match is strict. A correct answer phrased differently can still fail, so read the answers as well as the score. LangSmith's LLM-as-judge evaluator (below) is more tolerant of rephrasing.

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

The suite runs fully offline in about 10 seconds. A scripted fake model stands in for Claude, so it makes no API calls and sends nothing to LangSmith. It covers:

- **Retrieval** against the real local index: every test question finds its expected PDF, and the sub-agents' searches stay limited to their own PDFs
- **Scoring rules**
- **Cost math and monitoring storage**
- **Agents:** tool use, the approval flow (approve and reject), memory, email redaction, the call cap, delegation to sub-agents, and tool trimming
- **Every server endpoint**

## LangSmith

[LangSmith](https://smith.langchain.com/) is LangChain's hosted platform for tracing, evaluation and monitoring. Create an account and an API key (Settings → API Keys), then set these in `.env`:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=rag-demo
```

Restart the server. The sidebar shows "Tracing on".

### Tracing and monitoring
- **Traces:** every chat request becomes one trace in the `rag-demo` project, showing each LLM call, tool call and sub-agent with its tokens, cost and response time. Each trace is tagged with the mode (`rag`/`agent`/`multi`) and whether it came from chat or eval.
- **Threads:** agent traces carry `thread_id` metadata, so LangSmith groups each conversation under **Threads**.
- **User feedback:** 👍/👎 in the UI is attached to the trace as `user_score` (1/0), because the request id is also the LangSmith run id.
- **Configured in LangSmith, not in code:**
  - **Online evaluators:** rules that score new production traces automatically, e.g. an LLM judge on 10% of traces.
  - **Insights:** clusters traces into topics and failure patterns.
  - **Alerts:** on error rate, latency, cost or feedback score. Find these under the project's **Rules**, **Insights** and **Alerts** sections.

### Datasets and experiments

```bash
.venv/bin/python langsmith_eval.py --upload-only             # create the dataset (no Claude calls)
.venv/bin/python langsmith_eval.py --mode rag --limit 3      # run an experiment on 3 examples
.venv/bin/python langsmith_eval.py --mode agent --judge      # add the LLM-as-judge evaluator (extra tokens)
```

- **Dataset** `halvorsen-rag-demo` is created from `EVAL_CASES`. Add more examples in LangSmith, for example from a trace with **Add to dataset**.
- **Evaluators:**
  - **code-based** (`answer_correct`, `retrieval_hit`): free and deterministic
  - **LLM-as-judge** (`judge_correct`, Haiku): tolerant of rephrasing, but costs tokens and can be wrong. Check it against human labels with LangSmith's evaluator alignment.
- **Experiments** are named by mode and top-k (e.g. `agent-k3-...`). Compare them side by side under **Datasets & Experiments**: RAG vs Agent vs Multi-agent, or change `TOP_K` in `rag.py` and run again.
- **Offline vs online:** these experiments are *offline* evaluations against a fixed dataset. *Online* evaluators score live traffic (see above).

## Changing the knowledge base

Edit `DOCUMENTS` in `generate_docs.py` and run it again, or drop your own PDFs into `docs/`. If you change the facts, update `EVAL_CASES` in `evaluate.py` to match, and update `SUBAGENTS` in `agents.py` if files move between teams.

## Roadmap

These steps follow the four sections of the [LangChain Certified Agent Engineer exam guide](https://pub-dffa5f25c09c49488e245c43fcd4d0b5.r2.dev/exam_guide.pdf).

### 1. Build: agents and multi-agent
- [x] Single agent built with `create_agent`, with document search as a tool
- [x] Multi-agent built with `deepagents`: a supervisor that delegates to HR, Engineering and Sales sub-agents
- [x] Middleware: human approval for a sensitive tool, PII redaction, summarization, one custom middleware
- [x] Short-term memory (checkpointer, per thread)
- [ ] Long-term memory (store) across conversations
- [x] `AGENTS.md` instructions and a `SKILL.md` skill
- [x] File operations confined to a workspace with write permissions (not a full execution sandbox)

### 2. Test: evaluation
- [x] Local evaluation in the UI and command line, with cost estimates
- [x] LangSmith dataset from `EVAL_CASES`
- [x] Code-based evaluators and an LLM-as-judge evaluator
- [x] Experiments per mode (compare in LangSmith)
- [ ] Evaluator alignment and adding examples from traces (done in the LangSmith UI)
- [ ] Online evaluators (configured in LangSmith)

### 3. Deploy
- [ ] `langgraph.json`; run the agents with `langgraph dev` or on LangSmith Deployment
- [ ] Assistants (single and multi-agent configurations), threads and runs
- [ ] Custom auth: authentication vs authorization (e.g. only HR users can query HR documents)
- [ ] Secrets from environment variables; point the FastAPI UI at the deployment

### 4. Monitor
- [x] Local monitoring: tokens, cost, response time, errors and feedback per request
- [x] LangSmith tracing, with traces grouped into threads
- [x] 👍/👎 feedback in the UI, sent to LangSmith as `user_score`
- [ ] Online evals, Insights and alerts (configured in LangSmith)
