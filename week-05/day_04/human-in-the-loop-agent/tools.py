"""
Shared tool definitions: email, SQL, and weather.
"""

import os
import smtplib
from email.mime.text import MIMEText
import sqlite3
import requests
from langchain_core.tools import tool


# ---- Email tool (sensitive) ----
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send a real email to a recipient."""
    sender = os.environ["GMAIL_ADDRESS"]
    password = os.environ["GMAIL_APP_PASSWORD"]

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, [to], msg.as_string())

    return f"Real email sent to {to} with subject '{subject}'."


# ---- SQL tool (sensitive) ----
conn = sqlite3.connect("company.db", check_same_thread=False)
conn.execute("DROP TABLE IF EXISTS orders")
conn.execute(
    """
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        customer TEXT,
        amount REAL
    )
    """
)
conn.executemany(
    "INSERT INTO orders (customer, amount) VALUES (?, ?)",
    [("Alice", 250.0), ("Bob", 120.0), ("Charlie", 89.5)],
)
conn.commit()


@tool
def run_sql_query(query: str) -> str:
    """Run a real SQL query against the company SQLite database."""
    try:
        cur = conn.execute(query)
        if query.strip().lower().startswith("select"):
            rows = cur.fetchall()
            return str(rows)
        else:
            conn.commit()
            return f"Query executed. Rows affected: {cur.rowcount}"
    except Exception as e:
        return f"SQL error: {e}"


# ---- Weather tool (safe, no approval needed) ----
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    try:
        resp = requests.get(f"https://wttr.in/{city}?format=%C+%t", timeout=10)
        return f"Weather in {city}: {resp.text.strip()}"
    except Exception as e:
        return f"Weather error: {e}"


tools = [send_email, run_sql_query, get_weather]
tools_by_name = {t.name: t for t in tools}

# Tools that require human approval before execution (used by the
# human-in-the-loop-agent project; harmless to keep here too).
SENSITIVE_TOOLS = {"send_email", "run_sql_query"}
