# LangGraph Agent Demos

Two small LangGraph projects showing the same tool-using agent (email, SQL, weather) built two different ways.

## 📁 `basic-agent/`
A straightforward LangGraph agent. When the LLM decides to call a tool, it just runs — no confirmation step.

## 📁 `human-in-the-loop-agent/`
The same agent, but any *sensitive* tool call (sending an email, running a SQL query) triggers a `graph.interrupt()` and pauses for human approval before executing. Safe tools (like weather) still run automatically. Includes a small test harness (`test_agent.py`) covering both the approval flow and the no-approval-needed path.

## Setup

Each folder is self-contained — `cd` into whichever one you want and run:

```bash
pip install -r requirements.txt

export GEMINI_API_KEY="your-gemini-key"
export GMAIL_ADDRESS="you@gmail.com"          # only needed for send_email
export GMAIL_APP_PASSWORD="your-app-password" # only needed for send_email

python agent.py
```

## Notes

- Originally prototyped in Google Colab; Colab-specific auth (`userdata.get()`, interactive `getpass`) has been replaced with standard environment variables so it runs anywhere.
- Both projects share the same `tools.py` (email, SQL, weather) so behavior is directly comparable.
- The SQLite database (`company.db`) is created fresh each run with sample order data.
