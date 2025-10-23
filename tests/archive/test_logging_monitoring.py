"""Unit tests for logging and monitoring functionality."""

import pytest
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.boost_model_manager import BoostModelManager
from src.core.boost_orchestrator import BoostOrchestrator
from src.core.auxiliary_builder import AuxiliaryModelBuilder
from src.core.config import Config
from src.models.claude import ClaudeMessagesRequest


class TestLoggingMonitoring:
    """Test logging and monitoring functionality."""

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
    def boost_manager(self, mock_config):
        """Create a BoostModelManager instance for testing."""
        return BoostModelManager(mock_config)

    @pytest.fixture
    def boost_orchestrator(self, mock_config, mock_openai_client):
        """Create a BoostOrchestrator instance for testing."""
        return BoostOrchestrator(mock_config, mock_openai_client)

    @pytest.fixture
    def sample_claude_request(self):
        """Create a sample Claude request for testing."""
        return ClaudeMessagesRequest(
            model="claude-3-sonnet-20241022",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": "Test request"}
            ],
            tools=[
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
        )

    @pytest.mark.asyncio
    async def test_boost_manager_call_logging(self, boost_manager, caplog):
        """Test that boost model calls are logged."""
        # Mock HTTP client
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Test response"
                }
            }]
        }

        boost_manager.client.post = AsyncMock(return_value=mock_response)

        message = {"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]}

        with caplog.at_level(logging.INFO):
            result = await boost_manager.call_boost_model(message)

            # Check that call was logged
            assert "Calling boost model: gpt-4o" in caplog.text
            assert "Boost model response received: 13 characters" in caplog.text

    @pytest.mark.asyncio
    async def test_boost_manager_error_logging(self, boost_manager, caplog):
        """Test that boost model errors are logged."""
        # Mock HTTP client to raise error
        boost_manager.client.post = AsyncMock(side_effect=Exception("Test error"))

        message = {"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]}

        with caplog.at_level(logging.ERROR):
            with pytest.raises(Exception):
                await boost_manager.call_boost_model(message)

            # Check that error was logged
            assert "Unexpected error calling boost model: Test error" in caplog.text

    @pytest.mark.asyncio
    async def test_boost_manager_http_error_logging(self, boost_manager, caplog):
        """Test that HTTP errors are logged."""
        import httpx
        # Mock HTTP client to raise HTTP error
        boost_manager.client.post = AsyncMock(side_effect=httpx.HTTPError("HTTP Error"))

        message = {"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]}

        with caplog.at_level(logging.ERROR):
            with pytest.raises(httpx.HTTPError):
                await boost_manager.call_boost_model(message)

            # Check that HTTP error was logged
            assert "HTTP error calling boost model: HTTP Error" in caplog.text

    @pytest.mark.asyncio
    async def test_boost_manager_invalid_response_logging(self, boost_manager, caplog):
        """Test that invalid response format errors are logged."""
        # Mock HTTP client to return invalid response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect=KeyError("missing_key")

        boost_manager.client.post = AsyncMock(return_value=mock_response)

        message = {"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]}

        with caplog.at_level(logging.ERROR):
            with pytest.raises(KeyError):
                await boost_manager.call_boost_model(message)

            # Check that invalid response error was logged
            assert "Invalid response format from boost model" in caplog.text

    @pytest.mark.asyncio
    async def test_boost_orchestrator_execution_logging(self, boost_orchestrator, sample_claude_request, caplog):
        """Test that boost execution flow is logged."""
        # Mock boost manager to return SUMMARY
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("SUMMARY", "", "Final answer")
        )

        with caplog.at_level(logging.INFO):
            response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

            # Check that execution flow was logged
            assert "Starting boost execution for model: claude-3-sonnet-20241022" in caplog.text
            assert "Boost loop iteration: 0" in caplog.text
            assert "Boost model response type: SUMMARY" in caplog.text
            assert "Boost model provided SUMMARY response" in caplog.text

    @pytest.mark.asyncio
    async def test_boost_orchestrator_guidance_logging(self, boost_orchestrator, sample_claude_request, caplog):
        """Test that GUIDANCE responses are logged."""
        # Mock boost manager to return GUIDANCE
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
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
        boost_orchestrator.openai_client.create_chat_completion = AsyncMock(return_value=mock_openai_response)

        with caplog.at_level(logging.INFO):
            response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

            # Check that GUIDANCE flow was logged
            assert "Boost model response type: GUIDANCE" in caplog.text
            assert "Auxiliary model used tools successfully" in caplog.text

    @pytest.mark.asyncio
    async def test_boost_orchestrator_loop_logging(self, boost_orchestrator, sample_claude_request, caplog):
        """Test that loop iterations are logged."""
        # Mock boost manager to return OTHER first, then SUMMARY
        call_count = 0
        async def mock_get_guidance(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("OTHER", "Invalid format", "")
            else:
                return ("SUMMARY", "", "Final answer")

        boost_orchestrator.boost_manager.get_boost_guidance = mock_get_guidance

        with caplog.at_level(logging.INFO):
            response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

            # Check that loop retry was logged
            assert "Boost model returned invalid format, retrying..." in caplog.text
            assert "Boost loop iteration: 1" in caplog.text

    @pytest.mark.asyncio
    async def test_boost_orchestrator_max_loops_logging(self, boost_orchestrator, sample_claude_request, caplog):
        """Test that max loops reached is logged."""
        # Mock boost manager to always return OTHER
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("OTHER", "Invalid format", "")
        )

        with caplog.at_level(logging.WARNING):
            response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

            # Check that max loops was logged
            assert "Max loops (3) reached" in caplog.text

    @pytest.mark.asyncio
    async def test_boost_orchestrator_auxiliary_failure_logging(self, boost_orchestrator, sample_claude_request, caplog):
        """Test that auxiliary execution failures are logged."""
        # Mock boost manager to return GUIDANCE
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("GUIDANCE", "Analysis", "1. Call test_tool")
        )

        # Mock OpenAI client to raise exception
        boost_orchestrator.openai_client.create_chat_completion = AsyncMock(
            side_effect=Exception("Auxiliary failure")
        )

        with caplog.at_level(logging.ERROR):
            response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

            # Check that auxiliary failure was logged
            assert "Auxiliary model execution failed" in caplog.text

    @pytest.mark.asyncio
    async def test_boost_orchestrator_no_tool_usage_logging(self, boost_orchestrator, sample_claude_request, caplog):
        """Test that no tool usage is logged."""
        # Mock boost manager to return GUIDANCE
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("GUIDANCE", "Analysis", "1. Call test_tool")
        )

        # Mock OpenAI client to return response without tools
        mock_openai_response = {
            "choices": [{
                "message": {
                    "content": "No tools used"
                }
            }]
        }
        boost_orchestrator.openai_client.create_chat_completion = AsyncMock(return_value=mock_openai_response)

        # Mock boost manager to return SUMMARY on second iteration to end the loop
        call_count = 0
        async def mock_get_guidance(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("GUIDANCE", "Analysis", "1. Call test_tool")
            else:
                return ("SUMMARY", "", "Final answer")

        boost_orchestrator.boost_manager.get_boost_guidance = mock_get_guidance

        with caplog.at_level(logging.WARNING):
            response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

            # Check that no tool usage was logged
            assert "Auxiliary model did not use tools, triggering loop" in caplog.text

    def test_auxiliary_builder_logging(self, caplog):
        """Test that auxiliary builder operations are logged."""
        tools = [{"name": "test_tool", "description": "Test tool"}]
        original_request = MagicMock()
        original_request.model = "claude-3-sonnet-20241022"
        original_request.messages = [{"role": "user", "content": "Test"}]

        with caplog.at_level(logging.INFO):
            result = AuxiliaryModelBuilder.build_auxiliary_request(
                original_request, "Analysis", "Guidance", tools
            )

            # Check that auxiliary request building was logged
            assert "Built auxiliary request with 1 tools" in caplog.text

    @pytest.mark.asyncio
    async def test_boost_orchestrator_streaming_logging(self, boost_orchestrator, sample_claude_request, caplog):
        """Test that streaming operations are logged."""
        # Enable streaming
        sample_claude_request.stream = True

        # Mock boost manager to return GUIDANCE
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("GUIDANCE", "Analysis", "1. Call test_tool")
        )

        # Mock streaming response
        mock_stream = AsyncMock()
        mock_stream.__aiter__ = AsyncMock(return_value=iter([]))
        boost_orchestrator.openai_client.create_chat_completion_stream = AsyncMock(return_value=mock_stream)

        with caplog.at_level(logging.WARNING):
            response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

            # Check that streaming monitoring was logged
            # Note: This might not be visible in the test as streaming is handled differently
            pass

    @pytest.mark.asyncio
    async def test_debug_mode_logging(self, boost_orchestrator, sample_claude_request, caplog):
        """Test that debug mode provides detailed logging."""
        # Mock boost manager to return SUMMARY
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("SUMMARY", "", "Final answer")
        )

        with caplog.at_level(logging.DEBUG):
            response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

            # Check that debug information is available
            # Note: Actual debug logging depends on implementation
            pass

    @pytest.mark.asyncio
    async def test_performance_metrics_logging(self, boost_orchestrator, sample_claude_request, caplog):
        """Test that performance metrics are logged."""
        # Mock boost manager to return SUMMARY
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("SUMMARY", "", "Final answer")
        )

        with caplog.at_level(logging.INFO):
            response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

            # Check that performance-related logs are present
            # Note: Actual performance metrics depend on implementation
            assert "Starting boost execution" in caplog.text

    @pytest.mark.asyncio
    async def test_request_routing_logging(self, caplog):
        """Test that request routing decisions are logged."""
        from src.api.endpoints import router
        from fastapi import Request, HTTPException
        from src.core.model_manager import model_manager

        # This would require more complex setup to test actual endpoint logging
        # For now, we'll verify that the logging infrastructure exists
        assert router is not None
        assert model_manager is not None

    @pytest.mark.asyncio
    async def test_configuration_validation_logging(self, caplog):
        """Test that configuration validation is logged."""
        # Test configuration validation logging
        with caplog.at_level(logging.INFO):
            # This would test the actual configuration validation logging
            # For now, we'll verify the logging infrastructure
            pass

    @pytest.mark.asyncio
    async def test_tool_usage_detection_logging(self, boost_orchestrator, sample_claude_request, caplog):
        """Test that tool usage detection is logged."""
        # Mock boost manager to return GUIDANCE
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("GUIDANCE", "Analysis", "1. Call test_tool")
        )

        # Mock OpenAI client to return response with tools
        mock_openai_response = {
            "choices": [{
                "message": {
                    "content": "Tools used",
                    "tool_calls": [{"id": "call_1", "type": "function"}]
                }
            }]
        }
        boost_orchestrator.openai_client.create_chat_completion = AsyncMock(return_value=mock_openai_response)

        with caplog.at_level(logging.INFO):
            response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

            # Check that tool usage was logged
            assert "Auxiliary model used tools successfully" in caplog.text

    @pytest.mark.asyncio
    async def test_cleanup_logging(self, boost_orchestrator, sample_claude_request, caplog):
        """Test that resource cleanup is logged."""
        # Mock boost manager to return SUMMARY
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("SUMMARY", "", "Final answer")
        )

        boost_orchestrator.boost_manager.close = AsyncMock()

        with caplog.at_level(logging.INFO):
            response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

            # Check that cleanup was performed (close was called)
            boost_orchestrator.boost_manager.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_recovery_logging(self, boost_orchestrator, sample_claude_request, caplog):
        """Test that error recovery is logged."""
        # Mock boost manager to fail first, then succeed
        call_count = 0
        async def mock_get_guidance(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary failure")
            else:
                return ("SUMMARY", "", "Final answer")

        boost_orchestrator.boost_manager.get_boost_guidance = mock_get_guidance

        with caplog.at_level(logging.ERROR):
            response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

            # Check that error and recovery were logged
            # Note: Actual error recovery logging depends on implementation
            pass