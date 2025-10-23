from typing import List, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class LoopState:
    """Manages the state of the boost execution loop"""
    loop_count: int = 0
    max_loops: int = 3
    previous_attempts: List[str] = None
    current_guidance: str = ""
    current_analysis: str = ""

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