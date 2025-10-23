from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

@dataclass
class LoopState:
    """Manages the state of the boost execution loop"""
    loop_count: int = 0
    max_loops: int = 3
    previous_attempts: List[str] = None
    current_guidance: str = ""
    current_analysis: str = ""
    guidance_history: Set[str] = field(default_factory=set)
    analysis_history: Set[str] = field(default_factory=set)

    def __post_init__(self):
        if self.previous_attempts is None:
            self.previous_attempts = []

    def increment_loop(self) -> bool:
        """Increment loop count and return True if we can continue"""
        self.loop_count += 1
        return self.loop_count < self.max_loops

    def can_continue(self) -> bool:
        """Check if we can continue looping"""
        return self.loop_count < self.max_loops

    def add_attempt(self, attempt: str):
        """Add a failed attempt to the history"""
        self.previous_attempts.append(attempt)

    def get_context(self) -> Dict[str, Any]:
        """Get context for the next iteration"""
        return {
            "loop_count": self.loop_count,
            "previous_attempts": self.previous_attempts.copy(),
            "guidance": self.current_guidance,
            "analysis": self.current_analysis
        }

    def register_guidance(self, guidance: str) -> bool:
        """
        Track guidance values to detect duplicates.
        Returns True if the guidance is new, False if it has been seen.
        """
        normalized = (guidance or "").strip()
        self.current_guidance = normalized
        if not normalized:
            return False
        if normalized in self.guidance_history:
            return False
        self.guidance_history.add(normalized)
        return True

    def register_analysis(self, analysis: str) -> bool:
        """
        Track analysis values to detect duplicates.
        Returns True if the analysis is new, False if it has been seen.
        """
        normalized = (analysis or "").strip()
        self.current_analysis = normalized
        if not normalized:
            return False
        if normalized in self.analysis_history:
            return False
        self.analysis_history.add(normalized)
        return True

    def has_seen_guidance(self, guidance: str) -> bool:
        """Check if guidance content has already been processed."""
        normalized = (guidance or "").strip()
        return bool(normalized) and normalized in self.guidance_history

    def has_seen_analysis(self, analysis: str) -> bool:
        """Check if analysis content has already been processed."""
        normalized = (analysis or "").strip()
        return bool(normalized) and normalized in self.analysis_history
