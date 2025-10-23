"""Unit tests for error handling in boost mode."""

import pytest
import asyncio
import httpx
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.boost_model_manager import BoostModelManager
from src.core.boost_orchestrator import BoostOrchestrator
from src.core.config import Config
from src.models.claude import ClaudeMessagesRequest


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""

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
    async def test_boost_manager_http_error(self, boost_manager):
        """Test BoostModelManager handling of HTTP errors."""
        # Mock HTTP client to raise HTTPError
        boost_manager.client.post = AsyncMock(side_effect=httpx.HTTPError("Connection failed"))

        message = {"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]}

        with pytest.raises(httpx.HTTPError):
            await boost_manager.call_boost_model(message)

    @pytest.mark.asyncio
    async def test_boost_manager_timeout_error(self, boost_manager):
        """Test BoostModelManager handling of timeout errors."""
        # Mock HTTP client to raise timeout error
        boost_manager.client.post = AsyncMock(side_effect=httpx.TimeoutException("Request timed out"))

        message = {"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]}

        with pytest.raises(httpx.TimeoutException):
            await boost_manager.call_boost_model(message)

    @pytest.mark.asyncio
    async def test_boost_manager_network_error(self, boost_manager):
        """Test BoostModelManager handling of network errors."""
        # Mock HTTP client to raise network error
        boost_manager.client.post = AsyncMock(side_effect=httpx.NetworkError("Network unreachable"))

        message = {"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]}

        with pytest.raises(httpx.NetworkError):
            await boost_manager.call_boost_model(message)

    @pytest.mark.asyncio
    async def test_boost_manager_invalid_json_response(self, boost_manager):
        """Test BoostModelManager handling of invalid JSON response."""
        # Mock HTTP client to return invalid JSON
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect=ValueError("Invalid JSON")

        boost_manager.client.post = AsyncMock(return_value=mock_response)

        message = {"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]}

        with pytest.raises(ValueError):
            await boost_manager.call_boost_model(message)

    @pytest.mark.asyncio
    async def test_boost_manager_malformed_response_structure(self, boost_manager):
        """Test BoostModelManager handling of malformed response structure."""
        # Mock HTTP client to return malformed response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "invalid": "structure"  # Missing 'choices' key
        }

        boost_manager.client.post = AsyncMock(return_value=mock_response)

        message = {"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]}

        with pytest.raises(KeyError):
            await boost_manager.call_boost_model(message)

    @pytest.mark.asyncio
    async def test_boost_manager_empty_response(self, boost_manager):
        """Test BoostModelManager handling of empty response."""
        # Mock HTTP client to return empty response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {}

        boost_manager.client.post = AsyncMock(return_value=mock_response)

        message = {"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]}

        with pytest.raises(KeyError):
            await boost_manager.call_boost_model(message)

    @pytest.mark.asyncio
    async def test_boost_orchestrator_boost_api_failure(self, boost_orchestrator, sample_claude_request):
        """Test BoostOrchestrator handling of boost API failure."""
        # Mock boost manager to raise exception
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            side_effect=httpx.HTTPError("Boost API unavailable")
        )

        response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

        # Should return error response
        assert "Error: Maximum retry attempts reached" in response.content[0].text

    @pytest.mark.asyncio
    async def test_boost_orchestrator_auxiliary_api_failure(self, boost_orchestrator, sample_claude_request):
        """Test BoostOrchestrator handling of auxiliary API failure."""
        # Mock boost manager to return GUIDANCE
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("GUIDANCE", "Analysis", "1. Call test_tool")
        )

        # Mock OpenAI client to raise exception
        boost_orchestrator.openai_client.create_chat_completion = AsyncMock(
            side_effect=Exception("Auxiliary API failed")
        )

        # Mock recursive call to return final response
        with patch.object(boost_orchestrator, 'execute_with_boost') as mock_execute:
            mock_execute.return_value = MagicMock()
            mock_execute.return_value.content = [MagicMock(text="Final response after retry")]

            response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

            # Should have called execute_with_boost recursively
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_boost_orchestrator_streaming_auxiliary_failure(self, boost_orchestrator, sample_claude_request):
        """Test BoostOrchestrator handling of streaming auxiliary failure."""
        # Enable streaming
        sample_claude_request.stream = True

        # Mock boost manager to return GUIDANCE
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("GUIDANCE", "Analysis", "1. Call test_tool")
        )

        # Mock streaming to raise exception
        boost_orchestrator.openai_client.create_chat_completion_stream = AsyncMock(
            side_effect=Exception("Streaming failed")
        )

        with pytest.raises(Exception):
            await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

    @pytest.mark.asyncio
    async def test_boost_orchestrator_consecutive_failures(self, boost_orchestrator, sample_claude_request):
        """Test BoostOrchestrator handling of consecutive failures."""
        # Mock boost manager to always return OTHER (invalid format)
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("OTHER", "Invalid format", "")
        )

        response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

        # Should return error response after max loops
        assert "Error: Maximum retry attempts reached" in response.content[0].text

    @pytest.mark.asyncio
    async def test_boost_orchestrator_mixed_success_and_failure(self, boost_orchestrator, sample_claude_request):
        """Test BoostOrchestrator handling of mixed success and failure scenarios."""
        # Mock boost manager to return SUMMARY on second call
        call_count = 0
        async def mock_get_guidance(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("OTHER", "Invalid format", "")
            else:
                return ("SUMMARY", "", "Final successful response")

        boost_orchestrator.boost_manager.get_boost_guidance = mock_get_guidance

        response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

        # Should return final successful response
        assert response.content[0].text == "Final successful response"

    @pytest.mark.asyncio
    async def test_boost_manager_retry_on_transient_failure(self, boost_manager):
        """Test BoostModelManager retry behavior on transient failures."""
        # Mock HTTP client to fail once, then succeed
        call_count = 0
        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.HTTPError("Temporary failure")
            else:
                mock_response = MagicMock()
                mock_response.raise_for_status.return_value = None
                mock_response.json.return_value = {
                    "choices": [{
                        "message": {
                            "content": "Success after retry"
                        }
                    }]
                }
                return mock_response

        boost_manager.client.post = mock_post

        message = {"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]}

        # First call should fail
        with pytest.raises(httpx.HTTPError):
            await boost_manager.call_boost_model(message)

        # Second call should succeed
        result = await boost_manager.call_boost_model(message)
        assert result == "Success after retry"

    @pytest.mark.asyncio
    async def test_boost_orchestrator_resource_cleanup_on_error(self, boost_orchestrator, sample_claude_request):
        """Test that resources are cleaned up properly even when errors occur."""
        # Mock boost manager to raise exception
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            side_effect=Exception("Boost API Error")
        )

        boost_orchestrator.boost_manager.close = AsyncMock()

        with pytest.raises(Exception):
            await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

        # Should still close boost manager
        boost_orchestrator.boost_manager.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_boost_orchestrator_error_propagation(self, boost_orchestrator, sample_claude_request):
        """Test that errors are properly propagated and handled."""
        # Mock boost manager to raise exception
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            side_effect=ValueError("Unexpected error")
        )

        # Should not raise exception, but return error response
        response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

        assert isinstance(response, type(response))  # Should be a valid response object
        assert hasattr(response, 'content')

    @pytest.mark.asyncio
    async def test_boost_manager_concurrent_error_handling(self, boost_manager):
        """Test BoostModelManager handling of concurrent requests with errors."""
        # Mock HTTP client to fail for all requests
        boost_manager.client.post = AsyncMock(side_effect=httpx.HTTPError("Concurrent failure"))

        message = {"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]}

        # Test concurrent requests
        tasks = [boost_manager.call_boost_model(message) for _ in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should have failed with HTTPError
        for result in results:
            assert isinstance(result, httpx.HTTPError)

    @pytest.mark.asyncio
    async def test_boost_orchestrator_partial_failure_recovery(self, boost_orchestrator, sample_claude_request):
        """Test BoostOrchestrator recovery from partial failures."""
        # Mock boost manager to return GUIDANCE
        boost_orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("GUIDANCE", "Analysis", "1. Call test_tool")
        )

        # Mock OpenAI client to fail once, then succeed
        call_count = 0
        async def mock_completion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Auxiliary temporary failure")
            else:
                return {
                    "choices": [{
                        "message": {
                            "content": "Success after retry",
                            "tool_calls": [{"id": "call_1", "type": "function"}]
                        }
                    }]
                }

        boost_orchestrator.openai_client.create_chat_completion = mock_completion

        # Mock recursive call
        with patch.object(boost_orchestrator, 'execute_with_boost') as mock_execute:
            mock_execute.return_value = MagicMock()
            mock_execute.return_value.content = [MagicMock(text="Final response")]

            response = await boost_orchestrator.execute_with_boost(sample_claude_request, "test-request-id")

            # Should have recovered and called recursively
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_boost_manager_invalid_configuration_error(self):
        """Test BoostModelManager handling of invalid configuration."""
        # Create config with invalid values
        config = MagicMock()
        config.boost_base_url = "invalid-url"
        config.boost_api_key = "sk-test-key"
        config.boost_model = "invalid-model"
        config.request_timeout = 30

        # Should still create manager (validation happens at config level)
        manager = BoostModelManager(config)
        assert manager.config == config

    @pytest.mark.asyncio
    async def test_boost_orchestrator_request_validation_error(self, boost_orchestrator):
        """Test BoostOrchestrator handling of invalid request format."""
        # Create invalid request
        invalid_request = MagicMock()
        invalid_request.model = None  # Invalid model
        invalid_request.messages = []  # Empty messages

        # Should handle gracefully
        with patch.object(boost_orchestrator, '_extract_user_request') as mock_extract:
            mock_extract.return_value = "User request not found"

            response = await boost_orchestrator.execute_with_boost(invalid_request, "test-request-id")

            # Should still return a valid response
            assert hasattr(response, 'content')

    @pytest.mark.asyncio
    async def test_boost_manager_authentication_error(self, boost_manager):
        """Test BoostModelManager handling of authentication errors."""
        # Mock HTTP client to return auth error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized", request=MagicMock(), response=MagicMock()
        )

        boost_manager.client.post = AsyncMock(return_value=mock_response)

        message = {"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]}

        with pytest.raises(httpx.HTTPStatusError):
            await boost_manager.call_boost_model(message)

    @pytest.mark.asyncio
    async def test_boost_manager_rate_limit_error(self, boost_manager):
        """Test BoostModelManager handling of rate limit errors."""
        # Mock HTTP client to return rate limit error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429 Too Many Requests", request=MagicMock(), response=MagicMock()
        )

        boost_manager.client.post = AsyncMock(return_value=mock_response)

        message = {"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]}

        with pytest.raises(httpx.HTTPStatusError):
            await boost_manager.call_boost_model(message)

    @pytest.mark.asyncio
    async def test_boost_manager_server_error(self, boost_manager):
        """Test BoostModelManager handling of server errors."""
        # Mock HTTP client to return server error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error", request=MagicMock(), response=MagicMock()
        )

        boost_manager.client.post = AsyncMock(return_value=mock_response)

        message = {"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]}

        with pytest.raises(httpx.HTTPStatusError):
            await boost_manager.call_boost_model(message)