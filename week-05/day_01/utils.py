"""
utils.py

Small utility functions used across the project.
"""

import math


def estimate_tokens(messages):
    """
    Rough token estimation.

    Average:
        1 token ≈ 4 characters
    """

    chars = 0

    for role, text in messages:
        chars += len(text)

    return math.ceil(chars / 4)


def format_timestamp(dt):

    return dt.strftime("%Y-%m-%d %H:%M")


def message_statistics(messages):

    users = 0
    assistants = 0

    for role, _ in messages:

        if role == "user":
            users += 1
        else:
            assistants += 1

    return {
        "total": len(messages),
        "user": users,
        "assistant": assistants,
    }