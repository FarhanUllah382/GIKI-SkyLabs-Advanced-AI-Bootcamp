# Multi-Agent Research & Writing Workflow (LangGraph + Gemini + LangSmith)

A small multi-agent pipeline built with **LangGraph**, powered by **Google
Gemini** (`gemini-2.5-flash`), with run tracing sent to **LangSmith**.

A `supervisor` node routes work between two agents:

1. **Researcher** — breaks a topic down into 3–4 key factual points.
2. **Writer** — turns that research into a short (≤150 word) summary.

The supervisor loops `researcher → writer → end` based on who spoke last.

This project was refactored from a single Google Colab notebook into a
standard multi-file Python project.

## Project structure

```
ai_agent_project/
├── main.py            # Entry point — wires everything together and runs the graph
├── config.py           # Environment/config setup + LangSmith connectivity check
├── llm.py               # Builds the Gemini chat model
├── state.py             # Shared AgentState TypedDict
├── nodes.py             # researcher_node, writer_node, supervisor_node
├── graph.py              # Builds and compiles the LangGraph StateGraph
├── requirements.txt      # Python dependencies
├── .env.example          # Template for required environment variables
├── .gitignore
└── README.md
```

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure secrets

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```
LANGSMITH_API_KEY=your-langsmith-api-key-here
LANGCHAIN_PROJECT=My-first-project     # must already exist in your LangSmith workspace, case-sensitive
GEMINI_API_KEY=your-gemini-api-key-here
```

- Get a LangSmith API key from https://smith.langchain.com (Settings → API Keys).
- Get a Gemini API key from https://aistudio.google.com/apikey.
- Make sure `LANGCHAIN_PROJECT` matches an existing project name in your
  LangSmith workspace (or create one with that name).

### 3. Run it

```bash
python main.py
```

You should see the researcher and writer outputs printed to the console,
followed by a link to check the run in the LangSmith UI.

## Running in Google Colab

The original code pulled secrets via `google.colab.userdata.get(...)`
instead of a `.env` file. If you want to run this in Colab instead of
locally, only `config.py` needs to change:

```python
from google.colab import userdata

langsmith_api_key = userdata.get("langsmith_api_key")
gemini_api_key = userdata.get("GEMINI_API_KEY")
```

Everything else (`llm.py`, `state.py`, `nodes.py`, `graph.py`, `main.py`)
works unchanged — just upload the files to your Colab runtime (or clone
the repo) and run `main.py`, or paste each module's contents into its own
cell if you prefer to stay fully notebook-based.

## How it works

1. `config.py` sets `LANGCHAIN_TRACING_V2`, `LANGCHAIN_ENDPOINT`,
   `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT` **before** any
   LangChain/LangGraph object is created, since tracing config is read at
   construction time.
2. `verify_langsmith_connection()` makes a cheap `list_projects` call to
   confirm your API key/project are valid before spending any LLM calls.
3. `build_llm()` creates a `ChatGoogleGenerativeAI` instance.
4. `build_graph()` wires up a `StateGraph` with a supervisor that routes
   between `researcher` and `writer` nodes until the writer has spoken,
   then ends.
5. `main.py` streams a sample query ("Impact of AI agents on software
   engineering jobs") through the graph and prints each step's output.

## Customizing

- **Change the topic**: edit the `HumanMessage(content=...)` in `main.py`.
- **Change the model**: pass a different `model=` to `build_llm()` in
  `main.py` (e.g. `"gemini-2.5-pro"`).
- **Add more agents**: add a new node function in `nodes.py`, register it
  in `graph.py`, and extend the supervisor's routing logic.
