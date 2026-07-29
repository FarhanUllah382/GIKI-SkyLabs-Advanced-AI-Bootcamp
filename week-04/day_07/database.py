"""
database.py

Handles all SQLite operations for AI Chatbot Pro.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

DB_DIR = Path("data")
DB_DIR.mkdir(exist_ok=True)

DB_PATH = DB_DIR / "chatbot.db"


class ChatDatabase:

    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.create_tables()

    # ---------------------------------------------------
    # Database Schema
    # ---------------------------------------------------

    def create_tables(self):

        cursor = self.conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations(

            thread_id TEXT PRIMARY KEY,

            title TEXT,

            created_at TEXT,

            updated_at TEXT,

            summary TEXT DEFAULT '',

            total_messages INTEGER DEFAULT 0
        )
        """)

        self.conn.commit()

    # ---------------------------------------------------
    # Conversation Creation
    # ---------------------------------------------------

    def create_conversation(self, thread_id, title="New Conversation"):

        now = datetime.now().isoformat()

        cursor = self.conn.cursor()

        cursor.execute("""
        INSERT OR IGNORE INTO conversations(

            thread_id,
            title,
            created_at,
            updated_at

        )

        VALUES(?,?,?,?)

        """, (

            thread_id,
            title,
            now,
            now

        ))

        self.conn.commit()

    # ---------------------------------------------------
    # Update Title
    # ---------------------------------------------------

    def update_title(self, thread_id, new_title):

        cursor = self.conn.cursor()

        cursor.execute("""

        UPDATE conversations

        SET title=?,
            updated_at=?

        WHERE thread_id=?

        """,

        (

            new_title,
            datetime.now().isoformat(),
            thread_id

        ))

        self.conn.commit()

    # ---------------------------------------------------
    # Save Summary
    # ---------------------------------------------------

    def save_summary(self, thread_id, summary):

        cursor = self.conn.cursor()

        cursor.execute("""

        UPDATE conversations

        SET summary=?,
            updated_at=?

        WHERE thread_id=?

        """,

        (

            summary,
            datetime.now().isoformat(),
            thread_id

        ))

        self.conn.commit()

    # ---------------------------------------------------
    # Increment Message Count
    # ---------------------------------------------------

    def increment_messages(self, thread_id):

        cursor = self.conn.cursor()

        cursor.execute("""

        UPDATE conversations

        SET

            total_messages = total_messages + 1,

            updated_at = ?

        WHERE thread_id=?

        """,

        (

            datetime.now().isoformat(),

            thread_id

        ))

        self.conn.commit()

    # ---------------------------------------------------
    # Delete Conversation
    # ---------------------------------------------------

    def delete_conversation(self, thread_id):

        cursor = self.conn.cursor()

        cursor.execute("""

        DELETE FROM conversations

        WHERE thread_id=?

        """,

        (

            thread_id,

        ))

        self.conn.commit()

    # ---------------------------------------------------
    # Get One Conversation
    # ---------------------------------------------------

    def get_conversation(self, thread_id):

        cursor = self.conn.cursor()

        cursor.execute("""

        SELECT *

        FROM conversations

        WHERE thread_id=?

        """,

        (

            thread_id,

        ))

        row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    # ---------------------------------------------------
    # List All Conversations
    # ---------------------------------------------------

    def list_conversations(self):

        cursor = self.conn.cursor()

        cursor.execute("""

        SELECT *

        FROM conversations

        ORDER BY updated_at DESC

        """)

        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    # ---------------------------------------------------
    # Statistics
    # ---------------------------------------------------

    def statistics(self):

        cursor = self.conn.cursor()

        cursor.execute("""

        SELECT COUNT(*)

        FROM conversations

        """)

        total = cursor.fetchone()[0]

        cursor.execute("""

        SELECT SUM(total_messages)

        FROM conversations

        """)

        messages = cursor.fetchone()[0]

        if messages is None:
            messages = 0

        return {

            "conversations": total,

            "messages": messages

        }

    # ---------------------------------------------------
    # Close
    # ---------------------------------------------------

    def close(self):

        self.conn.close()