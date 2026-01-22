"""
Base classes for AI text detection.

Provides abstract interfaces and data classes for building detection methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DetectionResult:
    """Result from a single detection method."""

    score: float           # 0.0 (AI) to 1.0 (Human)
    confidence: float      # How confident the detector is (0.0 to 1.0)
    label: str             # "ai", "human", "uncertain"
    metrics: dict          # Method-specific metrics
    method_name: str       # Identifier for the detection method

    def __post_init__(self):
        # Clamp values to valid ranges
        self.score = max(0.0, min(1.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))

        # Validate label
        valid_labels = {"ai", "human", "uncertain"}
        if self.label not in valid_labels:
            raise ValueError(f"Invalid label: {self.label}. Must be one of {valid_labels}")


@dataclass
class EnsembleResult:
    """Result from ensemble detection combining multiple methods."""

    final_score: float           # Combined score (0.0 AI to 1.0 Human)
    confidence: float            # Overall confidence
    label: str                   # Final label
    individual_results: list     # List of DetectionResult from each method
    text_stats: dict = field(default_factory=dict)  # Basic text statistics

    def __post_init__(self):
        self.final_score = max(0.0, min(1.0, self.final_score))
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> dict:
        """Convert to dictionary format for API response."""
        return {
            "ensemble": {
                "score": round(self.final_score, 3),
                "label": self.label,
                "confidence": round(self.confidence, 3)
            },
            "methods": {
                result.method_name: {
                    "score": round(result.score, 3),
                    "confidence": round(result.confidence, 3),
                    "label": result.label,
                    "metrics": result.metrics
                }
                for result in self.individual_results
            },
            "text_stats": self.text_stats
        }


class BaseDetector(ABC):
    """Abstract base class for all detection methods."""

    @abstractmethod
    def detect(self, text: str) -> DetectionResult:
        """
        Analyze text and return detection result.

        Args:
            text: The text to analyze

        Returns:
            DetectionResult with score, confidence, and metrics
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the unique identifier for this detection method."""
        pass
