"""
Art Curation Engine Package

This package contains the AI-powered artwork curation system for emotional support.
It provides intelligent art recommendations based on user emotions and situations.
"""

# Import core components for easy access
from .core import (
    RAGSessionBrief,
    StageACollector,
    Step6LLMReranker,
    LangChainRAGSystem,
    Step6Prompts
)

__version__ = "1.0.0"
__author__ = "Art Curation Team"
__description__ = "AI-powered artwork curation system for emotional support"

__all__ = [
    "RAGSessionBrief",
    "StageACollector", 
    "Step6LLMReranker",
    "LangChainRAGSystem",
    "Step6Prompts"
]