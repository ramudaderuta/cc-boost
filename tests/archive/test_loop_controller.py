"""Unit tests for LoopController."""

import pytest
from src.core.loop_controller import LoopState


class TestLoopState:
    """Test LoopState functionality."""

    def test_init_default_values(self):
        """Test LoopState initialization with default values."""
        state = LoopState()

        assert state.loop_count == 0
        assert state.max_loops == 3
        assert state.previous_attempts == []
        assert state.current_guidance == ""
        assert state.current_analysis == ""

    def test_init_custom_values(self):
        """Test LoopState initialization with custom values."""
        state = LoopState(
            loop_count=1,
            max_loops=5,
            previous_attempts=["First attempt"],
            current_guidance="Test guidance",
            current_analysis="Test analysis"
        )

        assert state.loop_count == 1
        assert state.max_loops == 5
        assert state.previous_attempts == ["First attempt"]
        assert state.current_guidance == "Test guidance"
        assert state.current_analysis == "Test analysis"

    def test_init_none_previous_attempts(self):
        """Test LoopState initialization with None previous_attempts."""
        state = LoopState(previous_attempts=None)

        assert state.previous_attempts == []

    def test_init_empty_list_previous_attempts(self):
        """Test LoopState initialization with empty list previous_attempts."""
        state = LoopState(previous_attempts=[])

        assert state.previous_attempts == []

    def test_increment_loop_within_limit(self):
        """Test incrementing loop when within limit."""
        state = LoopState(loop_count=0, max_loops=3)

        can_continue = state.increment_loop()

        assert state.loop_count == 1
        assert can_continue is True

    def test_increment_loop_at_limit(self):
        """Test incrementing loop when at limit."""
        state = LoopState(loop_count=2, max_loops=3)

        can_continue = state.increment_loop()

        assert state.loop_count == 3
        assert can_continue is False

    def test_increment_loop_beyond_limit(self):
        """Test incrementing loop when beyond limit."""
        state = LoopState(loop_count=3, max_loops=3)

        can_continue = state.increment_loop()

        assert state.loop_count == 4
        assert can_continue is False

    def test_can_continue_true(self):
        """Test can_continue when loop count is less than max."""
        state = LoopState(loop_count=0, max_loops=3)
        assert state.can_continue() is True

        state.loop_count = 1
        assert state.can_continue() is True

        state.loop_count = 2
        assert state.can_continue() is True

    def test_can_continue_false(self):
        """Test can_continue when loop count equals or exceeds max."""
        state = LoopState(loop_count=3, max_loops=3)
        assert state.can_continue() is False

        state.loop_count = 4
        assert state.can_continue() is False

    def test_can_continue_equal_to_max(self):
        """Test can_continue when loop count equals max loops."""
        state = LoopState(loop_count=2, max_loops=2)
        assert state.can_continue() is False

    def test_add_attempt_empty_list(self):
        """Test adding attempt to empty previous_attempts."""
        state = LoopState()

        state.add_attempt("First attempt")

        assert state.previous_attempts == ["First attempt"]

    def test_add_attempt_existing_list(self):
        """Test adding attempt to existing previous_attempts."""
        state = LoopState(previous_attempts=["First attempt"])

        state.add_attempt("Second attempt")

        assert state.previous_attempts == ["First attempt", "Second attempt"]

    def test_add_attempt_multiple_attempts(self):
        """Test adding multiple attempts."""
        state = LoopState()

        state.add_attempt("Attempt 1")
        state.add_attempt("Attempt 2")
        state.add_attempt("Attempt 3")

        assert state.previous_attempts == ["Attempt 1", "Attempt 2", "Attempt 3"]

    def test_add_attempt_empty_string(self):
        """Test adding empty string attempt."""
        state = LoopState()

        state.add_attempt("")

        assert state.previous_attempts == [""]

    def test_add_attempt_long_string(self):
        """Test adding long string attempt."""
        long_attempt = "A" * 1000
        state = LoopState()

        state.add_attempt(long_attempt)

        assert state.previous_attempts == [long_attempt]

    def test_get_context_empty_state(self):
        """Test getting context from empty state."""
        state = LoopState()

        context = state.get_context()

        expected = {
            "loop_count": 0,
            "previous_attempts": [],
            "guidance": "",
            "analysis": ""
        }

        assert context == expected

    def test_get_context_populated_state(self):
        """Test getting context from populated state."""
        state = LoopState(
            loop_count=2,
            previous_attempts=["First attempt", "Second attempt"],
            current_guidance="Current guidance",
            current_analysis="Current analysis"
        )

        context = state.get_context()

        expected = {
            "loop_count": 2,
            "previous_attempts": ["First attempt", "Second attempt"],
            "guidance": "Current guidance",
            "analysis": "Current analysis"
        }

        assert context == expected

    def test_get_context_returns_copy(self):
        """Test that get_context returns a copy, not the original."""
        state = LoopState(previous_attempts=["Original"])

        context = state.get_context()
        context["previous_attempts"].append("Modified")

        # Original should not be modified
        assert state.previous_attempts == ["Original"]

    def test_increment_and_get_context(self):
        """Test incrementing loop and getting context."""
        state = LoopState()

        # Initial state
        context = state.get_context()
        assert context["loop_count"] == 0

        # Increment and check
        state.increment_loop()
        context = state.get_context()
        assert context["loop_count"] == 1

        # Add attempt and check
        state.add_attempt("Test attempt")
        context = state.get_context()
        assert context["previous_attempts"] == ["Test attempt"]

    def test_max_loops_boundary_conditions(self):
        """Test boundary conditions for max_loops."""
        # Test with max_loops = 0
        state = LoopState(max_loops=0)
        assert state.can_continue() is False
        assert state.increment_loop() is False

        # Test with max_loops = 1
        state = LoopState(max_loops=1)
        assert state.can_continue() is True
        assert state.increment_loop() is False

        # Test with max_loops = 10
        state = LoopState(max_loops=10)
        for i in range(10):
            assert state.can_continue() is (i < 10)  # can_continue() checks loop_count < max_loops
            if i < 9:
                assert state.increment_loop() is True
            else:
                assert state.increment_loop() is False

    def test_loop_state_immutability(self):
        """Test that loop state maintains proper immutability."""
        state = LoopState(
            loop_count=1,
            previous_attempts=["Attempt 1"]
        )

        # Get context and modify it
        context = state.get_context()
        context["loop_count"] = 999
        context["previous_attempts"].append("Modified")

        # Original state should be unchanged
        assert state.loop_count == 1
        assert state.previous_attempts == ["Attempt 1"]

    def test_complex_scenario(self):
        """Test a complex scenario with multiple operations."""
        state = LoopState(max_loops=2)

        # Initial state
        assert state.loop_count == 0
        assert state.can_continue() is True

        # First increment
        can_continue = state.increment_loop()
        assert state.loop_count == 1
        assert can_continue is True
        assert state.can_continue() is True

        # Add attempt
        state.add_attempt("First failed attempt")
        assert state.previous_attempts == ["First failed attempt"]

        # Second increment
        can_continue = state.increment_loop()
        assert state.loop_count == 2
        assert can_continue is False
        assert state.can_continue() is False

        # Add second attempt (should still work)
        state.add_attempt("Second failed attempt")
        assert state.previous_attempts == ["First failed attempt", "Second failed attempt"]

        # Get final context
        context = state.get_context()
        assert context["loop_count"] == 2
        assert context["previous_attempts"] == ["First failed attempt", "Second failed attempt"]
        assert context["guidance"] == ""
        assert context["analysis"] == ""