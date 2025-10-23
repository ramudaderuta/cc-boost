"""Unit tests for response processing and fallback logic."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.boost_orchestrator import BoostOrchestrator
from src.core.config import Config
from src.models.claude import ClaudeMessagesRequest, ClaudeMessageResponse


class TestResponseProcessing:
    """Test response processing and fallback functionality."""

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
        client.create_chat_completion_stream = MagicMock()
        return client

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
                {"role": "user", "content": "Read file /tmp/test.txt and analyze content"}
            ],
            tools=[
                {
                    "name": "read_file",
                    "description": "Read a file",
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
                }
            ]
        )

    @pytest.mark.asyncio
    async def test_extract_user_request_simple_text(self, boost_orchestrator):
        """Test extracting user request from simple text message."""
        messages = [
            {"role": "user", "content": "Simple request text"}
        ]

        result = boost_orchestrator._extract_user_request(messages)
        assert result == "Simple request text"

    @pytest.mark.asyncio
    async def test_extract_user_request_from_list_content(self, boost_orchestrator):
        """Test extracting user request from list content blocks."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First part"},
                    {"type": "text", "text": " second part"}
                ]
            }
        ]

        result = boost_orchestrator._extract_user_request(messages)
        assert result == "First part second part"

    @pytest.mark.asyncio
    async def test_extract_user_request_from_dict_content(self, boost_orchestrator):
        """Test extracting user request from dict content."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Request content"}
                ]
            }
        ]

        result = boost_orchestrator._extract_user_request(messages)
        assert result == "Request content"

    @pytest.mark.asyncio
    async def test_extract_user_request_multiple_messages(self, boost_orchestrator):
        """Test extracting user request from multiple messages."""
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "assistant", "content": "Previous response"},
            {"role": "user", "content": "Latest user request"}
        ]

        result = boost_orchestrator._extract_user_request(messages)
        assert result == "Latest user request"

    @pytest.mark.asyncio
    async def test_extract_user_request_no_user_message(self, boost_orchestrator):
        """Test extracting user request when no user message exists."""
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "assistant", "content": "Assistant response"}
        ]

        result = boost_orchestrator._extract_user_request(messages)
        assert result == "User request not found"

    @pytest.mark.asyncio
    async def test_extract_user_request_empty_content(self, boost_orchestrator):
        """Test extracting user request with empty content."""
        messages = [
            {"role": "user", "content": ""}
        ]

        result = boost_orchestrator._extract_user_request(messages)
        assert result == ""

    @pytest.mark.asyncio
    async def test_extract_user_request_complex_content(self, boost_orchestrator):
        """Test extracting user request with complex content structure."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read file "},
                    {"type": "text", "text": "/tmp/test.txt"},
                    {"type": "text", "text": " and analyze"}
                ]
            }
        ]

        result = boost_orchestrator._extract_user_request(messages)
        assert result == "Read file /tmp/test.txt and analyze"

    @pytest.mark.asyncio
    async def test_create_final_claude_response(self, boost_orchestrator, sample_claude_request):
        """Test creating final Claude response from boost SUMMARY."""
        final_text = "The capital of France is Paris."

        response = boost_orchestrator._create_final_claude_response(final_text, sample_claude_request)

        assert response.type == "message"
        assert response.role == "assistant"
        assert len(response.content) == 1
        assert response.content[0].type == "text"
        assert response.content[0].text == final_text
        assert response.model == sample_claude_request.model
        assert response.stop_reason == "end_turn"
        assert response.usage.input_tokens == 0
        assert response.usage.output_tokens == 0

    @pytest.mark.asyncio
    async def test_create_error_response(self, boost_orchestrator, sample_claude_request):
        """Test creating error response."""
        error_message = "Maximum retry attempts reached"

        response = boost_orchestrator._create_error_response(error_message, sample_claude_request)

        assert response.type == "message"
        assert response.role == "assistant"
        assert len(response.content) == 1
        assert response.content[0].type == "text"
        assert response.content[0].text == f"Error: {error_message}"
        assert response.model == sample_claude_request.model
        assert response.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_create_final_claude_response_long_text(self, boost_orchestrator, sample_claude_request):
        """Test creating final Claude response with long text."""
        final_text = "A" * 1000  # Long text

        response = boost_orchestrator._create_final_claude_response(final_text, sample_claude_request)

        assert response.content[0].text == final_text

    @pytest.mark.asyncio
    async def test_create_error_response_with_special_chars(self, boost_orchestrator, sample_claude_request):
        """Test creating error response with special characters."""
        error_message = "Error: API failed with status 500 (Internal Server Error)"

        response = boost_orchestrator._create_error_response(error_message, sample_claude_request)

        assert response.content[0].text == f"Error: {error_message}"

    @pytest.mark.asyncio
    async def test_execute_with_boost_summary_response(self, boost_orchestrator, sample_claude_request):
        """Test execute_with_boost with SUMMARY response."""
        # Mock boost manager to return SUMMARY
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("SUMMARY", "", "Final answer from boost model")
        )

        response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

        # Should return final Claude response
        assert isinstance(response, ClaudeMessageResponse)
        assert response.content[0].text == "Final answer from boost model"
        assert response.model == sample_claude_request.model

    @pytest.mark.asyncio
    async def test_execute_with_boost_guidance_response_non_streaming(self, boost_orchestrator, sample_claude_request):
        """Test execute_with_boost with GUIDANCE response and non-streaming."""
        # Mock boost manager to return GUIDANCE
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("GUIDANCE", "Analysis", "1. Call read_file with path: '/tmp/test.txt'")
        )

        # Mock OpenAI client to return response with tool usage
        mock_openai_response = {
            "choices": [{
                "message": {
                    "content": "File read successfully",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": "{\"path\": \"/tmp/test.txt\"}"
                            }
                        }
                    ]
                }
            }]
        }
        boost_orchestrator.openai_client.create_chat_completion = AsyncMock(return_value=mock_openai_response)

        response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

        # Should return converted Claude response
        assert isinstance(response, ClaudeMessageResponse)

    @pytest.mark.asyncio
    async def test_execute_with_boost_guidance_response_no_tool_usage(self, boost_orchestrator, sample_claude_request):
        """Test execute_with_boost when auxiliary model doesn't use tools."""
        # Mock boost manager to return GUIDANCE
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("GUIDANCE", "Analysis", "1. Call read_file with path: '/tmp/test.txt'")
        )

        # Mock OpenAI client to return response without tool usage
        mock_openai_response = {
            "choices": [{
                "message": {
                    "content": "I can help you without using any tools."
                }
            }]
        }
        boost_orchestrator.openai_client.create_chat_completion = AsyncMock(return_value=mock_openai_response)

        # Mock the recursive call to return final response
        with patch.object(boost_orchestrator, 'execute_with_boost') as mock_execute:
            mock_execute.return_value = MagicMock()
            mock_execute.return_value.content = [MagicMock(text="Final response after retry")]

            response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

            # Should have called execute_with_boost recursively
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_boost_other_response(self, boost_orchestrator, sample_claude_request):
        """Test execute_with_boost with OTHER response (invalid format)."""
        # Mock boost manager to return OTHER
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("OTHER", "Invalid format response", "")
        )

        # Mock the recursive call to return final response
        with patch.object(boost_orchestrator, 'execute_with_boost') as mock_execute:
            mock_execute.return_value = MagicMock()
            mock_execute.return_value.content = [MagicMock(text="Final response after retry")]

            response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

            # Should have called execute_with_boost recursively
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_boost_max_loops_reached(self, boost_orchestrator, sample_claude_request):
        """Test execute_with_boost when max loops are reached."""
        # Mock boost manager to always return OTHER (invalid format)
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("OTHER", "Invalid format response", "")
        )

        response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

        # Should return error response
        assert isinstance(response, ClaudeMessageResponse)
        assert "Error: Maximum retry attempts reached" in response.content[0].text

    @pytest.mark.asyncio
    async def test_execute_with_boost_auxiliary_execution_failure(self, boost_orchestrator, sample_claude_request):
        """Test execute_with_boost when auxiliary execution fails."""
        # Mock boost manager to return GUIDANCE
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("GUIDANCE", "Analysis", "1. Call read_file with path: '/tmp/test.txt'")
        )

        # Mock OpenAI client to raise exception
        boost_orchestrator.openai_client.create_chat_completion = AsyncMock(
            side_effect=Exception("API Error")
        )

        # Mock the recursive call to return final response
        with patch.object(boost_orchestrator, 'execute_with_boost') as mock_execute:
            mock_execute.return_value = MagicMock()
            mock_execute.return_value.content = [MagicMock(text="Final response after retry")]

            response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

            # Should have called execute_with_boost recursively
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_boost_streaming_guidance_response(self, boost_orchestrator, sample_claude_request):
        """Test execute_with_boost with GUIDANCE response and streaming."""
        # Modify request to enable streaming
        sample_claude_request.stream = True

        # Mock boost manager to return GUIDANCE
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("GUIDANCE", "Analysis", "1. Call read_file with path: '/tmp/test.txt'")
        )

        # Mock streaming response
        mock_stream = AsyncMock()
        mock_stream.__aiter__ = AsyncMock(return_value=iter([]))  # Empty stream for simplicity
        boost_orchestrator.openai_client.create_chat_completion_stream = AsyncMock(return_value=mock_stream)

        response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

        # Should return StreamingResponse
        from fastapi.responses import StreamingResponse
        assert isinstance(response, StreamingResponse)

    @pytest.mark.asyncio
    async def test_handle_non_streaming_auxiliary_with_tools(self, boost_orchestrator, sample_claude_request):
        """Test _handle_non_streaming_auxiliary when tools are used."""
        loop_state = MagicMock()
        loop_state.increment_loop.return_value = True

        auxiliary_request = {"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]}

        mock_openai_response = {
            "choices": [{
                "message": {
                    "content": "Tools used successfully",
                    "tool_calls": [{"id": "call_1", "type": "function"}]
                }
            }]
        }
        boost_orchestrator.openai_client.create_chat_completion = AsyncMock(return_value=mock_openai_response)

        with patch('src.core.boost_orchestrator.convert_openai_to_claude_response') as mock_convert:
            mock_convert.return_value = MagicMock()

            response = await boost_orchestrator._handle_non_streaming_auxiliary(
                auxiliary_request, sample_claude_request, "test-id", loop_state
            )

            # Should convert and return response
            mock_convert.assert_called_once_with(mock_openai_response, sample_claude_request)
            assert response == mock_convert.return_value

    @pytest.mark.asyncio
    async def test_handle_non_streaming_auxiliary_without_tools(self, boost_orchestrator, sample_claude_request):
        """Test _handle_non_streaming_auxiliary when no tools are used."""
        loop_state = MagicMock()
        loop_state.increment_loop.return_value = False  # Max loops reached

        auxiliary_request = {"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]}

        mock_openai_response = {
            "choices": [{
                "message": {
                    "content": "No tools used"
                }
            }]
        }
        boost_orchestrator.openai_client.create_chat_completion = AsyncMock(return_value=mock_openai_response)

        with patch.object(boost_orchestrator, '_create_error_response') as mock_error:
            mock_error.return_value = MagicMock()

            response = await boost_orchestrator._handle_non_streaming_auxiliary(
                auxiliary_request, sample_claude_request, "test-id", loop_state
            )

            # Should return error response
            mock_error.assert_called_once()
            loop_state.increment_loop.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_boost_cleanup(self, boost_orchestrator, sample_claude_request):
        """Test that boost manager is cleaned up after execution."""
        # Mock boost manager to return SUMMARY
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("SUMMARY", "", "Final answer")
        )

        boost_orchestrator.boost_manager.close = AsyncMock()

        await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

        # Should close boost manager
        boost_orchestrator.boost_manager.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_boost_cleanup_on_exception(self, boost_orchestrator, sample_claude_request):
        """Test that boost manager is cleaned up even when exception occurs."""
        # Mock boost manager to raise exception
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            side_effect=Exception("Boost API Error")
        )

        boost_orchestrator.boost_manager.close = AsyncMock()

        response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

        # Should return error response after retries and ensure cleanup
        assert "Error:" in response.content[0].text
        boost_orchestrator.boost_manager.close.assert_awaited_once()

        # Should still close boost manager
        boost_orchestrator.boost_manager.close.assert_called_once()
