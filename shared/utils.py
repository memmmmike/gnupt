"""
Common text processing utilities for AI text detection.
"""

import re
from typing import Optional


def count_valid_chars(text: str) -> int:
    """Count alphanumeric characters in text."""
    valid_chars = re.findall(r"[a-zA-Z0-9]+", text)
    return sum(len(x) for x in valid_chars)


def is_valid_text(text: str, min_chars: int = 100) -> tuple[bool, str]:
    """
    Check if text meets minimum requirements for analysis.

    Args:
        text: Text to validate
        min_chars: Minimum number of alphanumeric characters

    Returns:
        Tuple of (is_valid, error_message)
    """
    total_valid = count_valid_chars(text)

    if total_valid < min_chars:
        return False, f"Text too short: {total_valid} chars (minimum {min_chars})"

    return True, ""


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences.

    Args:
        text: Text to split

    Returns:
        List of sentence strings
    """
    lines = re.split(r'(?<=[.?!][ \[\(])|(?<=\n)\s*', text)
    lines = [line for line in lines if line and line.strip()]
    return lines


def clean_sentence(sentence: str) -> str:
    """
    Clean a sentence for processing.

    Removes leading/trailing whitespace and handles edge cases.
    """
    sentence = sentence.strip()

    # Remove leading newlines or spaces
    while sentence and sentence[0] in "\n ":
        sentence = sentence[1:]

    # Remove trailing newlines or spaces
    while sentence and sentence[-1] in "\n ":
        sentence = sentence[:-1]

    return sentence


def tokenize_words(text: str) -> list[str]:
    """Split text into lowercase words."""
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())


def has_alphanumeric(text: str) -> bool:
    """Check if text contains any alphanumeric characters."""
    return re.search(r"[a-zA-Z0-9]+", text) is not None


def remove_citations(text: str) -> str:
    """Remove citation markers like [1], [2], etc."""
    return re.sub(r"\[[0-9]+\]", "", text)
