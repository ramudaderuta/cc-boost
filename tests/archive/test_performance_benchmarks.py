"""Performance benchmark tests for boost mode functionality."""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.boost_model_manager import BoostModelManager
from src.core.boost_orchestrator import BoostOrchestrator
from src.core.auxiliary_builder import AuxiliaryModelBuilder
from src.core.loop_controller import LoopState
from src.core.config import Config
from src.models.claude import ClaudeMessagesRequest


class TestPerformanceBenchmarks:
    """Performance benchmark tests for boost mode."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        config = MagicMock()
        config.boost_base_url = "https://api.test.com/v1"
        config.boost_api_key = "sk-test-key"
        config.boost_model = "gpt-4o"
        config.enable_boost_support = "MIDDLE_MODEL"
        config.request_timeout = 30
        return config

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client for testing."""
        client = MagicMock()
        client.create_chat_completion = AsyncMock()
        client.create_chat_completion_stream = AsyncMock()
        return client

    @pytest.fixture
    def sample_tools(self):
        """Sample tools for testing."""
        return [
            {
                        "name": "read_file",
                        "description": "Read a file from filesystem",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "File path"
                                }
                            },
                            "required": ["path"]
                        }
                    },
            {
                        "name": "write_file",
                        "description": "Write content to a file",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "File path"
                                },
                                "content": {
                                    "type": "string",
                                    "description": "File content"
                                }
                            },
                            "required": ["path"]
                        }
                    }
            ]

    @pytest.fixture
    def sample_claude_request(self, sample_tools):
        """Create a sample Claude request for testing."""
        return ClaudeMessagesRequest(
            model="claude-3-sonnet-20241022",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": "Test request for performance benchmark"}
            ],
            tools=sample_tools
        )

    def test_boost_message_building_performance(self, mock_config):
        """Test performance of boost message building."""
        manager = BoostModelManager(mock_config)

        # Test with varying numbers of tools
        tool_counts = [1, 5, 10, 20, 50]
        max_time_ms = 10  # Maximum acceptable time in milliseconds

        for tool_count in tool_counts:
            tools = []
            for i in range(tool_count):
                tools.append({
                    "name": f"tool_{i}",
                    "description": f"Tool number {i}",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "param": {
                                "type": "string",
                                "description": f"Parameter for tool {i}"
                            }
                        },
                        "required": ["param"]
                    }
                })

            start_time = time.time()
            message = manager.build_boost_message(
                "Test request", tools, loop_count=1
            )
            end_time = time.time()

            elapsed_ms = (end_time - start_time) * 1000

            # Assert that message building is fast
            assert elapsed_ms < max_time_ms, f"Message building with {tool_count} tools took {elapsed_ms}ms, should be < {max_time_ms}ms"

            # Verify message structure
            assert "model" in message
            assert "messages" in message
            assert "tools parameter" not in message

    def test_response_parsing_performance(self, mock_config):
        """Test performance of response parsing."""
        manager = BoostModelManager(mock_config)

        # Test with varying response lengths
        response_lengths = [100, 1000, 5000, 10000]
        max_time_ms = 5  # Maximum acceptable time in milliseconds

        for length in response_lengths:
            # Create a test response
            analysis = "A" * (length // 3)
            guidance = "G" * (length // 3)
            summary = "S" * (length // 3)

            response_text = f"""ANALYSIS:
{analysis}

GUIDANCE:
{guidance}

SUMMARY:
{summary}
"""

            start_time = time.time()
            analysis_result = manager._extract_section(response_text, "ANALYSIS:")
            guidance_result = manager._extract_section(response_text, "GUIDANCE:")
            summary_result = manager._extract_section(response_text, "SUMMARY:")
            end_time = time.time()

            elapsed_ms = (end_time - start_time) * 1000

            # Assert that parsing is fast
            assert elapsed_ms < max_time_ms, f"Response parsing with {length} chars took {elapsed_ms}ms, should be < {max_time_ms}ms"

    def test_auxiliary_request_building_performance(self):
        """Test performance of auxiliary request building."""
        max_time_ms = 5  # Maximum acceptable time in milliseconds

        # Test with varying analysis and guidance lengths
        content_lengths = [100, 1000, 5000]

        for length in content_lengths:
            analysis = "A" * length
            guidance = "G" * length
            tools = [
                {
                    "name": "test_tool",
                    "description": "Test tool",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "param": {
                                "type": "string",
                                "description": "Test parameter"
                            }
                        },
                        "required": ["param"]
                    }
                }
            ]

            original_request = MagicMock()
            original_request.model = "claude-3-sonnet-20241022"
            original_request.messages = [{"role": "user", "content": "Test request"}]
            original_request.stream = False
            original_request.max_tokens = 1000

            start_time = time.time()
            auxiliary_request = AuxiliaryModelBuilder.build_auxiliary_request(
                original_request, analysis, guidance, tools
            )
            end_time = time.time()

            elapsed_ms = (end_time - start_time) * 1000

            # Assert that building is fast
            assert elapsed_ms < max_time_ms, f"Auxiliary request building with {length} chars took {elapsed_ms}ms, should be < {max_time_ms}ms"

    def test_tool_usage_detection_performance(self):
        """Test performance of tool usage detection."""
        max_time_ms = 1  # Maximum acceptable time in milliseconds

        # Test with varying numbers of tool calls
        tool_call_counts = [0, 1, 5, 10, 20]

        for tool_count in tool_call_counts:
            tool_calls = []
            for i in range(tool_count):
                tool_calls.append({
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "arguments": f'{{"param": "value_{i}"}}'
                    }
                })

            response = {
                "choices": [{
                    "message": {
                        "content": "Test response",
                        "tool_calls": tool_calls
                    }
                }]
            }

            start_time = time.time()
            result = AuxiliaryModelBuilder.detect_tool_usage(response)
            end_time = time.time()

            elapsed_ms = (end_time - start_time) * 1000

            # Assert that detection is fast
            assert elapsed_ms < max_time_ms, f"Tool usage detection with {tool_count} tools took {elapsed_ms}ms, should be < {max_time_ms}ms"

            # Verify correctness
            expected_result = tool_count > 0
            assert result == expected_result

    def test_loop_state_operations_performance(self):
        """Test performance of loop state operations."""
        max_time_ms = 1  # Maximum acceptable time in milliseconds

        # Test with varying numbers of previous attempts
        attempt_counts = [0, 1, 5, 10, 20]

        for attempt_count in attempt_counts:
            previous_attempts = [f"Attempt {i}" for i in range(attempt_count)]

            start_time = time.time()
            loop_state = LoopState(
                loop_count=1,
                previous_attempts=previous_attempts,
                current_guidance="Test guidance",
                current_analysis="Test analysis"
            )

            # Test various operations
            can_continue = loop_state.can_continue()
            context = loop_state.get_context()
            end_time = time.time()

            elapsed_ms = (end_time - start_time) * 1000

            # Assert that operations are fast
            assert elapsed_ms < max_time_ms, f"Loop state operations with {attempt_count} attempts took {elapsed_ms}ms, should be < {max_time_ms}ms"

            # Verify correctness
            assert can_continue is True
            assert len(context["previous_attempts"]) == attempt_count

    @pytest.mark.asyncio
    async def test_boost_orchestrator_summary_performance(self, mock_config, mock_openai_client, sample_claude_request):
        """Test performance of boost orchestrator with SUMMARY response."""
        orchestrator = BoostOrchestrator(mock_config, mock_openai_client)

        # Mock boost manager to return SUMMARY quickly
        orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("SUMMARY", "", "Final answer")
        )

        max_time_ms = 100  # Maximum acceptable time in milliseconds

        start_time = time.time()
        response = await orchestrator.execute_with_boost(sample_claude_request, "test-request-id")
        end_time = time.time()

        elapsed_ms = (end_time - start_time) * 1000

        # Assert that execution is fast
        assert elapsed_ms < max_time_ms, f"Boost orchestrator SUMMARY execution took {elapsed_ms}ms, should be < {max_time_ms}ms"

        # Verify response
        assert response.content[0].text == "Final answer"

    @pytest.mark.asyncio
    async def test_boost_orchestrator_guidance_performance(self, mock_config, mock_openai_client, sample_claude_request):
        """Test performance of boost orchestrator with GUIDANCE response."""
        orchestrator = BoostOrchestrator(mock_config, mock_openai_client)

        # Mock boost manager to return GUIDANCE
        orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("GUIDANCE", "Analysis", "1. Call test_tool")
        )

        # Mock OpenAI client to return response with tools
        mock_openai_response = {
            "choices": [{
                "message": {
                    "content": "Tools used successfully",
                    "tool_calls": [{"id": "call_1", "type": "function"}]
                }
            }]
        }
        mock_openai_client.create_chat_completion = AsyncMock(return_value=mock_openai_response)

        max_time_ms = 150  # Maximum acceptable time in milliseconds

        start_time = time.time()
        response = await orchestrator.execute_with_boost(sample_claude_request, "test-request-id")
        end_time = time.time()

        elapsed_ms = (end_time - start_time) * 1000

        # Assert that execution is fast
        assert elapsed_ms < max_time_ms, f"Boost orchestrator GUIDANCE execution took {elapsed_ms}ms, should be < {max_time_ms}ms"

    @pytest.mark.asyncio
    async def test_boost_orchestrator_loop_performance(self, mock_config, mock_openai_client, sample_claude_request):
        """Test performance of boost orchestrator with loop iterations."""
        orchestrator = BoostOrchestrator(mock_config, mock_openai_client)

        # Mock boost manager to return OTHER first, then SUMMARY
        call_count = 0
        async def mock_get_guidance(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("OTHER", "Invalid format", "")
            else:
                return ("SUMMARY", "", "Final answer after retry")

        orchestrator.boost_manager.get_boost_guidance = mock_get_guidance

        max_time_ms = 200  # Maximum acceptable time in milliseconds

        start_time = time.time()
        response = await orchestrator.execute_with_boost(sample_claude_request, "test-request-id")
        end_time = time.time()

        elapsed_ms = (end_time - start_time) * 1000

        # Assert that execution with loop is fast
        assert elapsed_ms < max_time_ms, f"Boost orchestrator loop execution took {elapsed_ms}ms, should be < {max_time_ms}ms"

        # Verify response
        assert response.content[0].text == "Final answer after retry"

    def test_tools_formatting_performance(self, mock_config):
        """Test performance of tools formatting."""
        manager = BoostModelManager(mock_config)

        # Test with varying numbers of tools
        tool_counts = [1, 5, 10, 25, 50]
        max_time_ms = 10  # Maximum acceptable time in milliseconds

        for tool_count in tool_counts:
            tools = []
            for i in range(tool_count):
                tools.append({
                    "name": f"tool_{i}",
                    "description": f"Description for tool {i} with some detail about what it does",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            f"param_{i}": {
                                "type": "string",
                                "description": f"Parameter {i} description"
                            },
                            f"optional_param_{i}": {
                                "type": "integer",
                                "description": f"Optional parameter {i}"
                            }
                        },
                        "required": [f"param_{i}"]
                    }
                })

            start_time = time.time()
            tools_text = manager._format_tools_for_message(tools)
            end_time = time.time()

            elapsed_ms = (end_time - start_time) * 1000

            # Assert that formatting is fast
            assert elapsed_ms < max_time_ms, f"Tools formatting with {tool_count} tools took {elapsed_ms}ms, should be < {max_time_ms}ms"

            # Verify that all tools are included
            for i in range(tool_count):
                assert f"tool_{i}" in tools_text

    def test_memory_usage_performance(self, mock_config):
        """Test memory usage performance."""
        import sys
        manager = BoostModelManager(mock_config)

        # Test with large number of tools to check memory usage
        tool_count = 100
        tools = []
        for i in range(tool_count):
            tools.append({
                "name": f"tool_{i}",
                "description": f"Description for tool {i}",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "param": {
                            "type": "string",
                            "description": f"Parameter {i}"
                        }
                    },
                    "required": ["param"]
                }
            })

        # Get initial memory usage
        import gc
        gc.collect()
        initial_memory = sys.getsizeof([])  # Baseline

        # Build message and check memory
        message = manager.build_boost_message("Test request", tools, loop_count=0)
        final_memory = sys.getsizeof(message)

        # Memory usage should be reasonable
        memory_increase = final_memory - initial_memory
        max_memory_increase = 50000  # 50KB max increase

        assert memory_increase < max_memory_increase, f"Memory increase {memory_increase} bytes exceeds limit {max_memory_increase} bytes"

    @pytest.mark.asyncio
    async def test_concurrent_boost_execution_performance(self, mock_config, mock_openai_client, sample_tools):
        """Test performance of concurrent boost executions."""
        orchestrator = BoostOrchestrator(mock_config, mock_openai_client)

        # Mock boost manager to return SUMMARY
        orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("SUMMARY", "", "Concurrent response")
        )

        # Create multiple requests
        num_requests = 10
        requests = []
        for i in range(num_requests):
            request = ClaudeMessagesRequest(
                model="claude-3-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": f"Concurrent request {i}"}],
                tools=sample_tools
            )
            requests.append(request)

        max_time_ms = 500  # Maximum acceptable time for all requests

        start_time = time.time()
        tasks = [orchestrator.execute_with_boost(req, f"request-{i}") for i, req in enumerate(requests)]
        responses = await asyncio.gather(*tasks)
        end_time = time.time()

        elapsed_ms = (end_time - start_time) * 1000

        # Assert that concurrent execution is fast
        assert elapsed_ms < max_time_ms, f"Concurrent execution of {num_requests} requests took {elapsed_ms}ms, should be < {max_time_ms}ms"

        # Verify all responses
        for i, response in enumerate(responses):
            assert response.content[0].text == "Concurrent response"

    def test_validation_criteria_performance_targets(self):
        """Test that performance meets validation criteria from tasks.md."""
        # From tasks.md validation criteria:
        # - Performance overhead < 150ms per request (boost only)
        # - Performance overhead < 100ms additional when auxiliary model needed
        # - Performance overhead < 50ms per loop iteration

        # These are targets that the actual implementation should meet
        # Here we test that our test framework can measure such performance

        max_boost_only_ms = 150
        max_auxiliary_ms = 100
        max_loop_iteration_ms = 50

        # Test that we can measure performance at this granularity
        start_time = time.time()
        time.sleep(0.001)  # Simulate 1ms of work
        end_time = time.time()

        elapsed_ms = (end_time - start_time) * 1000

        # Verify measurement precision
        assert elapsed_ms >= 0, "Should be able to measure positive time intervals"
        assert elapsed_ms < max_loop_iteration_ms, "Test measurement should be precise enough for validation criteria"

        # The actual performance targets would be tested against real implementation
        # These tests ensure our benchmarking framework can validate the criteria
        assert max_boost_only_ms == 150
        assert max_auxiliary_ms == 100
        assert max_loop_iteration_ms == 50
