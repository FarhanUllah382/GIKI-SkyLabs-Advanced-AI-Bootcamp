# CRAG — Corrective Retrieval-Augmented Generation

A self-correcting RAG pipeline built with **LangGraph** and **Claude (Anthropic)**.
Instead of blindly trusting retrieved documents, the graph *grades* them,
falls back to a live web search when the vectorstore comes up short, and
*grades its own answer* for hallucination and relevance before returning it.

## How it works

```
                 ┌───────────┐
                 │ retrieve  │
                 └─────┬─────┘
                       │
                 ┌─────▼──────────┐
                 │ grade_documents │
                 └─────┬───────────┘
             relevant  │  irrelevant doc found
          ┌────────────┴────────────┐
          ▼                         ▼
     ┌─────────┐            ┌────────────────┐
     │ generate│◄───────┐   │ transform_query │
     └────┬────┘        │   └───────┬────────┘
          │             │           ▼
          │             │     ┌────────────┐
          │             │     │ web_search │
          │             │     └─────┬──────┘
          │             └───────────┘
          ▼
  grade_generation
   ├─ not grounded  → generate (retry)
   ├─ not useful     → transform_query (rewrite & re-search)
   └─ useful         → END
```

1. **retrieve** — pull candidate chunks from a local Chroma vectorstore.
2. **grade_documents** — an LLM grader scores each chunk for relevance to
   the question. Any irrelevant chunk flips `web_search_needed`.
3. **transform_query** — if triggered, rewrites the question into a
   better standalone web-search query.
4. **web_search** — fetches live results via Tavily and appends them to
   the document set.
5. **generate** — answers using only the currently held documents.
6. **grade_generation** — checks the answer is (a) grounded in the
   documents (not hallucinated) and (b) actually answers the question.
   Ungrounded answers trigger a retry; unhelpful-but-grounded answers
   trigger a fresh web search; good answers end the graph.
7. **retries** — capped by `CRAG_MAX_RETRIES` (default 3) so the graph
   always terminates instead of looping forever.

## Project structure

```
crag_project/
├── main.py                 # CLI entry point
├── requirements.txt
├── .env.example
└── src/
    ├── config.py            # env loading, shared LLM instance, tunables
    ├── state.py              # GraphState TypedDict
    ├── graders.py             # structured-output relevance/hallucination/answer graders
    ├── retriever.py            # Chroma vectorstore + retriever setup
    ├── nodes.py                # retrieve / grade_documents / transform_query / web_search / generate
    ├── edges.py                 # decide_to_generate / grade_generation routing logic
    └── graph.py                  # wires nodes + edges into the compiled LangGraph app
```

## What was added vs. the original prototype

The original single-file script referenced a `retriever` object that was
never defined, and a `retries` field with no path to actually increment
it. This version fixes both:

- **`src/retriever.py`** builds a real Chroma vectorstore (HuggingFace
  sentence-transformer embeddings) from a small seed set of URLs — swap
  in your own document loader for production use.
- **`generate`** now increments `state["retries"]` on every call, so the
  `MAX_RETRIES` bailout in `grade_generation` actually works.
- **`.env.example` / `src/config.py`** centralize API keys and fail fast
  with a clear error if `ANTHROPIC_API_KEY` or `TAVILY_API_KEY` is missing,
  instead of failing deep inside a node.
- **`main.py`** gives you a runnable CLI instead of just a compiled graph
  object sitting unused.

## Setup

```bash
git clone <your-repo-url>
cd crag_project
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and add your ANTHROPIC_API_KEY and TAVILY_API_KEY
```

## Usage

```bash
python main.py "What is agent memory in LLM applications?"
```

Or interactively:

```bash
python main.py
Question: What is agent memory in LLM applications?
```

## Configuration

All tunables live in `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | required |
| `TAVILY_API_KEY` | — | required, powers `web_search` fallback |
| `CRAG_MODEL_NAME` | `claude-sonnet-4-5` | model used for graders + generation |
| `CRAG_TEMPERATURE` | `0` | LLM temperature |
| `CRAG_MAX_RETRIES` | `3` | max `generate` retries before forced exit |
| `CRAG_WEB_SEARCH_RESULTS` | `3` | number of Tavily results to pull |

## Extending

- **Swap the corpus**: edit `SOURCE_URLS` / `load_documents()` in
  `src/retriever.py`, or point it at PDFs, a database, or an existing
  index.
- **Tune the relevance threshold**: `grade_documents` currently flags a
  web search if *any* document is irrelevant — change it to a majority
  vote or a minimum relevant-doc count if that's too aggressive.
- **Swap the web search provider**: replace `TavilySearchResults` in
  `src/nodes.py` with any other LangChain search tool.
- **Persist the vectorstore**: `build_retriever()` already writes to
  `.chroma_db/`; reuse it across runs instead of re-embedding every
  startup by checking if the directory already has data before calling
  `Chroma.from_documents`.

## Notes

- Graders use `with_structured_output` so responses come back as typed
  Pydantic objects (`binary_score: "yes" | "no"`) rather than free text,
  which keeps the conditional routing in `src/edges.py` deterministic.
- The graph is intentionally small and readable — no memory/checkpointing
  is wired in, so each `app.invoke(...)` call is a fresh, stateless run.
  Add a LangGraph checkpointer if you need multi-turn conversation memory.
