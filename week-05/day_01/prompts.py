"""
prompts.py

Contains every prompt used by the chatbot.

Advantages:
- Easy to update prompts
- Easy to experiment
- Keeps graph.py clean
"""

# ============================================================
# Main System Prompt
# ============================================================

SYSTEM_PROMPT = """
You are ChatBot Pro, a highly capable AI assistant.

Your responsibilities:

• Answer accurately.
• Be honest if you are uncertain.
• Use previous conversation whenever relevant.
• Maintain context across the discussion.
• Format code inside Markdown code blocks.
• Explain difficult topics step-by-step.
• Be friendly and professional.
• Avoid hallucinating facts.
• When asked to compare, create tables whenever appropriate.
• Prefer concise answers unless the user explicitly requests detail.
"""

# ============================================================
# Conversation Title Prompt
# ============================================================

TITLE_PROMPT = """
Generate a short title for this conversation.

Rules:

- Maximum 5 words
- No quotation marks
- No emojis
- No punctuation at the end
- Return ONLY the title

Conversation:

{conversation}
"""

# ============================================================
# Conversation Summary Prompt
# ============================================================

SUMMARY_PROMPT = """
Summarize the following conversation.

Requirements:

- Keep important facts.
- Preserve user preferences.
- Preserve unfinished tasks.
- Preserve names if mentioned.
- Remove repetition.
- Maximum 200 words.

Conversation:

{conversation}
"""

# ============================================================
# Memory Update Prompt
# ============================================================

MEMORY_PROMPT = """
Extract long-term memory from this conversation.

Return ONLY information worth remembering later.

Examples:

- User preferences
- Favorite programming language
- Learning goals
- Ongoing projects
- Personal profile

Do NOT include temporary discussion.

Conversation:

{conversation}
"""

# ============================================================
# Rename Conversation Prompt
# ============================================================

RENAME_PROMPT = """
Generate a better title.

Current Title:

{title}

Conversation:

{conversation}

Return only the improved title.
"""

# ============================================================
# Reflection Prompt
# ============================================================

REFLECTION_PROMPT = """
Review your previous answer.

Check:

- Was anything incorrect?
- Was anything missing?
- Can the explanation be improved?

If yes,
produce an improved version.

Otherwise return the original answer.
"""