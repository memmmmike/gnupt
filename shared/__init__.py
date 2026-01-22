"""
GnuPT Shared Library

Common components for AI text detection used by GPTZero and DetectGPT.
"""

from .base import BaseDetector, DetectionResult, EnsembleResult
from .gpt2_scorer import GPT2Scorer, get_scorer
from .utils import (
    count_valid_chars,
    is_valid_text,
    split_sentences,
    clean_sentence,
    tokenize_words,
    has_alphanumeric,
    remove_citations,
)

__all__ = [
    # Base classes
    "BaseDetector",
    "DetectionResult",
    "EnsembleResult",
    # GPT-2 scoring
    "GPT2Scorer",
    "get_scorer",
    # Utilities
    "count_valid_chars",
    "is_valid_text",
    "split_sentences",
    "clean_sentence",
    "tokenize_words",
    "has_alphanumeric",
    "remove_citations",
]
