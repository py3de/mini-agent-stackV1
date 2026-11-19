# Mini Agent Stack

[![tests](https://github.com/py3de/mini-agent-stackV1/actions/workflows/tests.yml/badge.svg)](https://github.com/py3de/mini-agent-stackV1/actions/workflows/tests.yml)

**A model-agnostic LangChain / LangGraph reference stack for internal AI tools.**

Companies that use LLMs across many internal tools often end up with every tool wired up differently:
different SDKs, different prompts, no shared tracing, no way to compare models. This project is a
small version of the alternative: **one shared foundation** that every tool is built on.

The demo tool is a **meeting-notes → CRM agent**: paste raw notes from a customer call, and the agent
finds the contact and deal in HubSpot, logs a meeting note, moves the deal stage if the call changed
it, looks up answers to the customer's open questions in a knowledge base (RAG), and returns a
validated, structured report with a follow-up email. **No CRM change happens without a human
approving it**, and every decision ends up in an audit trail.

| What | How |
|---|---|
| Model-agnostic LLM layer | One `get_llm()` for Gemini, OpenAI, Anthropic and Ollama, switched by config |
| Agent | Explicit LangGraph `StateGraph`: tool calling loop, governance gate, structured-output node |
| Governance | Read/write tool permissions (deny by default), human approval gates via LangGraph interrupts, read-only mode, audit trail, escalation |
| Structured outputs | Pydantic `MeetingReport`, identical across providers |
| Tools | HubSpot CRM tools (mock CRM by default, real HubSpot API with one env var) |
| RAG | Markdown knowledge base, section-level chunks, vector store, model-agnostic embeddings |
| MCP | The same tools exposed as an MCP server, written once and reused |
| API / webhooks | FastAPI with a secret-protected webhook and a token-protected approval endpoint |
| Resilience | Automatic fallback models plus retries with backoff for overload / rate-limit errors |
| Tracing | LangSmith, enabled with environment variables |
| Evals | Same test cases run across providers, scored on what actually changed in the CRM |
| Tests | Full agent runs tested with a scripted fake model, no API key needed |

## Architecture

```mermaid
flowchart LR
    subgraph Entry points
        CLI[CLI]
        API[FastAPI<br/>/agent/run<br/>/webhooks/meeting-notes]
        MCPC[MCP clients<br/>Claude Desktop, IDEs]
        EVAL[Eval harness]
    end

    subgraph Core
        LLM["get_llm()<br/>model-agnostic layer"]
        AGENT["LangGraph agent<br/>agent ⇄ gate → report"]
        GOV["Governance<br/>permissions · approvals<br/>audit trail · escalation"]
        SCHEMA[MeetingReport<br/>Pydantic schema]
        TOOLS["Tool registry<br/>search_contacts · search_deals<br/>create_note · update_deal_stage<br/>search_knowledge_base"]
    end

    subgraph Providers
        G[Gemini]
        O[OpenAI]
        A[Anthropic]
        L[Ollama, local]
    end

    HS[(HubSpot CRM<br/>mock or live)]
    KB[(Knowledge base<br/>vector store)]
    LS[LangSmith tracing]

    CLI --> AGENT
    API --> AGENT
    EVAL --> AGENT
    MCPC --> MCP[MCP server] --> TOOLS
    AGENT --> LLM --> G & O & A & L
    AGENT --> TOOLS --> HS
    TOOLS --> KB
    AGENT --> SCHEMA
    AGENT --> GOV
    AGENT -.-> LS
```

The agent graph itself:

```mermaid
graph TD;
    __start__([start]) --> agent;
    agent -. tool calls .-> gate;
    gate -. write action: interrupt .-> human{{human approves / rejects}};
    human -. resume .-> gate;
    gate --> agent;
    gate -. repeated errors: escalate .-> report;
    agent -. done or step limit .-> report;
    report --> __end__([end]);
```

## Quickstart

```bash
git clone https://github.com/py3de/mini-agent-stackV1.git
cd mini-agent-stackV1
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
```

Add **one** API key to `.env`. The cheapest way to start is a free Gemini key from
[Google AI Studio](https://aistudio.google.com/apikey), or install [Ollama](https://ollama.com) and run
`ollama pull llama3.1` (no key needed). Model names change often, so you can override them with
`MODEL_GOOGLE`, `MODEL_OPENAI` and so on.

```bash
agent-stack providers                                  # which providers are ready
agent-stack run --notes-file examples/nordwerk.txt     # run the agent on the default provider
agent-stack run --provider ollama --notes-file examples/nordwerk.txt
pytest                                                 # tests run without any API key
```

HubSpot runs in **mock mode** by default (an in-memory CRM with demo contacts and deals). To use a
real account, create a free HubSpot developer test account with a private app that has CRM
read/write scopes, then set `HUBSPOT_MODE=live` and `HUBSPOT_ACCESS_TOKEN`.

## Using it

### Governance: approvals, permissions and audit trail

Every tool is classified as **read** or **write** (`governance.py`). Unknown tools count as writes, so a
new tool can never change data silently. Three modes (`GOVERNANCE_MODE` or `--mode`):

| Mode | Reads | Writes |
|---|---|---|
| `approve` (default) | run freely | the run **pauses** and waits for a human to approve or reject each action |
| `read_only` | run freely | always **blocked**, safe for investigation |
| `auto` | run freely | run without approval, for tests, evals and trusted batch jobs |

In the CLI you are asked per action:

```text
$ agent-stack run --notes-file examples/nordwerk.txt
The agent wants to make these changes:
  - create_note({"contact_id": "101", "body": "Budget approved ..."})
    Approve? [y/N] y
  - update_deal_stage({"deal_id": "9001", "stage": "contractsent"})
    Approve? [y/N] n

Tool calls (governance decision):
  [  read  ] search_contacts({"query": "Lena Fischer"})
  [approved] create_note({"contact_id": "101", ...})
  [REJECTED] update_deal_stage({"deal_id": "9001", "stage": "contractsent"})
```

Rejected or blocked actions are never executed. The model is told about it, so it continues
without retrying, and the report lists only changes that really happened.

**How the pause works:** the `gate` node calls LangGraph's `interrupt()` *before* running any tool in
that turn. The run's state is saved by a checkpointer under its run ID and resumed later with
`Command(resume=...)`, in the same process or through the API. Nothing in the gate has side effects
before the interrupt, so it's safe that LangGraph re-runs the node on resume.

**Audit trail:** every run start, approval request, decision, tool call (with permission and decision),
escalation and run end is appended to `audit/audit.jsonl`:

```bash
agent-stack audit --run-id 3f2a9c1b7e44
```

**Escalation:** after `ESCALATE_AFTER_ERRORS` (default 2) consecutive tool errors, or when the step limit
is reached, the run stops with status `escalated` and is flagged for human review instead of letting
the model keep guessing.


### HTTP API and webhook

```bash
uvicorn agent_stack.api:app --reload
```

```bash
curl -X POST localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"notes": "Call with Jonas Weber (AlpenTech): they will stay with their current provider.", "provider": "google"}'

# External systems (e.g. a meeting recorder) call the webhook with a shared secret:
curl -X POST localhost:8000/webhooks/meeting-notes \
  -H "X-Webhook-Secret: change-me" -H "Content-Type: application/json" \
  -d '{"notes": "..."}'
```

In `approve` mode the response is `"status": "awaiting_approval"` with the pending actions. A reviewer
approves or rejects them, all at once or per action ID, with the approver token:

```bash
curl -X POST localhost:8000/runs/<run_id>/decision \
  -H "X-Approver-Token: change-me-approver" -H "Content-Type: application/json" \
  -d '{"decision": {"call_abc": "approve", "call_def": "reject"}}'

curl localhost:8000/runs/<run_id>/audit      # the run's audit trail
```

Callers can request a *stricter* mode (`"mode": "read_only"`) but never `auto`; that is only
available through server configuration.

Interactive docs are available at `http://localhost:8000/docs`.

### MCP server

The tools are also available to any MCP client. MCP clients have no approval gate, so the server is
**read-only by default**: it only exposes read tools, and every call is audited. Set
`MCP_ALLOW_WRITES=true` to expose write tools as well. For Claude Desktop, add this to
`claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mini-agent-stack-crm": {
      "command": "/absolute/path/to/mini-agent-stack/.venv/bin/python",
      "args": ["-m", "agent_stack.mcp_server"]
    }
  }
}
```

### Knowledge base (RAG)

`knowledge/*.md` holds company documents (security & hosting, services, contracts & pricing). Each
`##` section becomes one chunk, so every retrieved result is a self-contained answer with a clear
source like `security-and-hosting.md > Data residency`. When meeting notes contain a customer
question ("do you host in the EU?"), the agent calls `search_knowledge_base` and the follow-up email
answers it with facts instead of guesses.

Embeddings are model-agnostic too (`EMBEDDINGS_PROVIDER` = google, openai or ollama). The vector store
is LangChain's `InMemoryVectorStore`, rebuilt on start; moving to pgvector / Supabase only changes
`build_vector_store()` in `rag.py`. To use your own documents, drop markdown files into `knowledge/`.

### Tracing with LangSmith

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in `.env`. Every run, LLM call and tool call
then shows up in LangSmith under the project `mini-agent-stack`, tagged with the provider.

### Evals: comparing models

```bash
python evals/run_evals.py                          # every provider with a key
python evals/run_evals.py --providers google ollama
```

Each case in `evals/cases.json` resets the CRM, runs the agent and then checks **what actually
happened in the CRM**, not just what the model says it did:

- **Schema:** a valid `MeetingReport` came back
- **Note:** a note was logged on the right contact (and nothing for unknown contacts)
- **Stage:** the right deal moved to the right stage, and no other deal was touched
- **IDs:** the report's deal ID matches reality (no invented IDs)
- **KB:** the knowledge base was searched when the customer asked a question

Results, including latency and number of tool calls, are written to `evals/results/results.md`.

<!-- After running the evals, paste evals/results/results.md here and add 2-3 sentences on the trade-offs you saw. -->

## Adding a new tool

1. Write a function with `@tool` in a module under `src/agent_stack/tools/`. The docstring is what
   the model reads, so describe **when** to use the tool, not only what it does.
2. Return errors as text instead of raising, so the model can recover.
3. Add it to a toolset in `src/agent_stack/tools/__init__.py`.

The agent, the API and the MCP server pick it up automatically. Add an eval case if the tool
changes data.

## Design decisions

- **One LLM entry point.** Tools never import a provider SDK. `get_llm()` uses LangChain's
  `init_chat_model`, so switching or adding a provider is configuration, and every model gets
  tool calling and structured output through the same interface.
- **An explicit `StateGraph` instead of a prebuilt agent.** The loop, the step limit and the final
  structured-output node are visible and testable. The step limit protects against runaway tool loops.
- **Separate structured-output step.** Mixing "call tools" and "return JSON" in one call is fragile on
  some providers. A dedicated `report` node makes the output schema reliable on all of them.
- **Write tools once, use them everywhere.** The MCP server wraps the existing LangChain tools instead of
  reimplementing them, so the agent and MCP clients can never drift apart.
- **Evals check side effects, not claims.** A model can say "I updated the deal" without doing it. The
  evals inspect the CRM state after each run.
- **Resilience in two layers.** Each model call first falls back to backup models (`LLM_FALLBACKS`), and
  graph nodes retry transient errors (503 overload, 504 deadline exceeded, 429 rate limits) with exponential backoff. Permanent
  errors such as a wrong key or model name fail fast with a readable message instead of being retried.
  Provider SDKs are set to fail fast (1 retry, 120 s timeout) so an overloaded model hands over to the
  backup in seconds. With SDK defaults, one overloaded run took over 6 minutes.
- **Section-level chunking for RAG.** Company documents are short and well structured, so one chunk per
  `##` section keeps answers complete and sources readable. Fixed-size chunking would cut answers in half.
- **Governance in the graph, not in the prompt.** Telling a model "ask before you change things" is a
  request; a gate node that physically cannot run a write tool without a recorded human decision is a
  control. Permissions are deny-by-default, rejections are fed back to the model, and every decision is
  audited, so "who changed this deal and who approved it?" always has an answer.
- **Mock-first.** The mock CRM and the scripted fake model mean everything can be developed and tested
  without accounts, keys or cost.

## Project structure

```
src/agent_stack/
  llm.py             model-agnostic LLM layer (provider registry)
  agent.py           LangGraph agent: agent ⇄ gate → report, start / resume runs
  governance.py      permissions, governance modes, audit trail
  schemas.py         structured outputs (MeetingReport)
  rag.py             knowledge base: chunking, embeddings, vector store
  tools/             tool registry, CRM tools, knowledge-base tool
  hubspot_client.py  mock and live HubSpot backends
  api.py             FastAPI app and webhook
  mcp_server.py      MCP server exposing the same tools
  cli.py             command line interface
evals/               eval cases and the provider comparison harness
tests/               tests with a scripted fake model (no API keys)
examples/            sample meeting notes
knowledge/           markdown documents for the knowledge base (RAG)
```

## Roadmap

- [x] Human-in-the-loop approval gates, tool permissions, audit trail and escalation
- [x] RAG over a knowledge base (vector store) so the agent can answer customer questions in follow-up emails
- [ ] Postgres / Supabase checkpointer, so paused approvals survive restarts (today: in memory)
- [ ] Small Lovable front end on top of the API
- [ ] More eval cases and LLM-as-judge scoring for the summary and email quality

## License

MIT
