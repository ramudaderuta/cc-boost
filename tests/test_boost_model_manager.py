"""Unit tests for BoostModelManager."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from src.core.boost_model_manager import BoostModelManager
from src.core.config import Config


class TestBoostModelManager:
    """Test BoostModelManager functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        config = MagicMock()
        config.boost_base_url = "https://api.test.com/v1"
        config.boost_api_key = "sk-test-boost-key"
        config.boost_model = "gpt-4o"
        config.boost_wrapper_template = None
        config.request_timeout = 30
        return config

    @pytest.fixture
    def boost_manager(self, mock_config):
        """Create a BoostModelManager instance for testing."""
        return BoostModelManager(mock_config)

    def test_init(self, mock_config):
        """Test BoostModelManager initialization."""
        manager = BoostModelManager(mock_config)
        assert manager.config == mock_config
        assert str(manager.client.base_url) == mock_config.boost_base_url
        assert manager.client.timeout == mock_config.request_timeout

    def test_get_default_wrapper_template(self, boost_manager):
        """Test default wrapper template generation."""
        template = boost_manager._get_default_wrapper_template()

        # Check that template contains required placeholders
        assert "{loop_count}" in template
        assert "{previous_attempts}" in template
        assert "{user_request}" in template
        assert "{tools_text}" in template

        # Check that template contains required sections
        assert "SUMMARY:" in template
        assert "ANALYSIS:" in template
        assert "GUIDANCE:" in template
        assert "FORMAT 1" in template
        assert "FORMAT 2" in template
        assert "FORMAT 3" in template

    def test_format_tools_for_message_no_tools(self, boost_manager):
        """Test formatting tools when no tools are provided."""
        tools_text = boost_manager._format_tools_for_message([])
        assert tools_text == "No tools available"

    def test_format_tools_for_message_single_tool(self, boost_manager):
        """Test formatting a single tool."""
        tools = [{
            "name": "search_web",
            "description": "Search the web for information",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }]

        tools_text = boost_manager._format_tools_for_message(tools)
        assert "search_web: Search the web for information" in tools_text
        assert "query: string (required) - Search query" in tools_text

    def test_format_tools_for_message_multiple_tools(self, boost_manager):
        """Test formatting multiple tools."""
        tools = [
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
            },
            {
                "name": "write_file",
                "description": "Write to a file",
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

        tools_text = boost_manager._format_tools_for_message(tools)
        assert "read_file: Read a file" in tools_text
        assert "write_file: Write to a file" in tools_text
        assert "path: string (required) - File path" in tools_text
        assert "content: string (optional) - File content" in tools_text

    def test_format_tools_for_message_no_schema(self, boost_manager):
        """Test formatting tools without input schema."""
        tools = [{
            "name": "simple_tool",
            "description": "A simple tool without schema"
        }]

        tools_text = boost_manager._format_tools_for_message(tools)
        assert "simple_tool: A simple tool without schema" in tools_text

    def test_build_boost_message_default_template(self, boost_manager):
        """Test building boost message with default template."""
        user_request = "What is the weather like?"
        tools = [{
            "name": "get_weather",
            "description": "Get weather information",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location"
                    }
                },
                "required": ["location"]
            }
        }]

        message = boost_manager.build_boost_message(user_request, tools, loop_count=1)

        # Check message structure
        assert message["model"] == "gpt-4o"
        assert len(message["messages"]) == 1
        assert message["messages"][0]["role"] == "user"
        assert "temperature" in message
        assert "max_tokens" in message

        # Check that tools are embedded in content
        content = message["messages"][0]["content"]
        assert "Current ReAct Loop: 1" in content
        assert "User Request: What is the weather like?" in content
        assert "get_weather: Get weather information" in content
        assert "tools parameter" not in message  # Should NOT have tools parameter

    def test_build_boost_message_custom_template(self, boost_manager):
        """Test building boost message with custom template."""
        boost_manager.config.boost_wrapper_template = "Custom: {user_request} with {tools_text}"

        user_request = "Test request"
        tools = []

        message = boost_manager.build_boost_message(user_request, tools)

        content = message["messages"][0]["content"]
        assert content == "Custom: Test request with No tools available"

    def test_build_boost_message_with_previous_attempts(self, boost_manager):
        """Test building boost message with previous attempts."""
        user_request = "Test request"
        tools = []
        previous_attempts = ["First attempt failed", "Second attempt failed"]

        message = boost_manager.build_boost_message(user_request, tools, loop_count=2, previous_attempts=previous_attempts)

        content = message["messages"][0]["content"]
        assert "Current ReAct Loop: 2" in content
        assert "- First attempt failed" in content
        assert "- Second attempt failed" in content

    @pytest.mark.asyncio
    async def test_call_boost_model_success(self, boost_manager):
        """Test successful boost model call."""
        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Test response content"
                }
            }]
        }

        boost_manager.client.post = AsyncMock(return_value=mock_response)

        message = {"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]}
        result = await boost_manager.call_boost_model(message)

        assert result == "Test response content"
        boost_manager.client.post.assert_called_once_with("/chat/completions", json=message)

    @pytest.mark.asyncio
    async def test_call_boost_model_http_error(self, boost_manager):
        """Test boost model call with HTTP error."""
        boost_manager.client.post = AsyncMock(side_effect=httpx.HTTPError("API Error"))

        message = {"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]}

        with pytest.raises(httpx.HTTPError):
            await boost_manager.call_boost_model(message)

    @pytest.mark.asyncio
    async def test_call_boost_model_invalid_response(self, boost_manager):
        """Test boost model call with invalid response format."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "invalid": "response format"  # Missing choices structure
        }

        boost_manager.client.post = AsyncMock(return_value=mock_response)

        message = {"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]}

        with pytest.raises(KeyError):
            await boost_manager.call_boost_model(message)

    def test_extract_section_summary(self, boost_manager):
        """Test extracting SUMMARY section."""
        text = """
SUMMARY:
This is the summary content
It can span multiple lines

ANALYSIS:
Some analysis content
---
"""

        result = boost_manager._extract_section(text, "SUMMARY:")
        assert result == "This is the summary content\nIt can span multiple lines\n"

    def test_extract_section_analysis(self, boost_manager):
        """Test extracting ANALYSIS section."""
        text = """
ANALYSIS:
This is the analysis content
It has multiple lines

GUIDANCE:
Some guidance content
---
"""

        result = boost_manager._extract_section(text, "ANALYSIS:")
        assert result == "This is the analysis content\nIt has multiple lines\n"

    def test_extract_section_guidance(self, boost_manager):
        """Test extracting GUIDANCE section."""
        text = """
GUIDANCE:
This is the guidance content
It spans multiple lines

---
More content
"""

        result = boost_manager._extract_section(text, "GUIDANCE:")
        assert result == "This is the guidance content\nIt spans multiple lines"

    def test_extract_section_not_found(self, boost_manager):
        """Test extracting section that doesn't exist."""
        text = """
ANALYSIS:
Some analysis

SUMMARY:
Some summary
---
"""

        result = boost_manager._extract_section(text, "GUIDANCE:")
        assert result is None

    def test_extract_section_empty_content(self, boost_manager):
        """Test extracting section with empty content."""
        text = """
ANALYSIS:

GUIDANCE:
Some guidance
---
"""

        result = boost_manager._extract_section(text, "ANALYSIS:")
        assert result == ""

    @pytest.mark.asyncio
    async def test_get_boost_guidance_summary_response(self, boost_manager):
        """Test get_boost_guidance with SUMMARY response."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "SUMMARY:\nThe capital of France is Paris."
                }
            }]
        }

        boost_manager.client.post = AsyncMock(return_value=mock_response)

        response_type, analysis, guidance = await boost_manager.get_boost_guidance(
            "What is the capital of France?", []
        )

        assert response_type == "SUMMARY"
        assert analysis == ""
        assert guidance == "The capital of France is Paris."

    @pytest.mark.asyncio
    async def test_get_boost_guidance_guidance_response(self, boost_manager):
        """Test get_boost_guidance with GUIDANCE response."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": """ANALYSIS:
The user wants to read a file and analyze its content.

GUIDANCE:
1. Call read_file with path: '/tmp/test.txt'
2. Analyze the content and provide summary"""
                }
            }]
        }

        boost_manager.client.post = AsyncMock(return_value=mock_response)

        response_type, analysis, guidance = await boost_manager.get_boost_guidance(
            "Read and analyze /tmp/test.txt", []
        )

        assert response_type == "GUIDANCE"
        assert analysis == "The user wants to read a file and analyze its content."
        assert guidance == "1. Call read_file with path: '/tmp/test.txt'\n2. Analyze the content and provide summary"

    @pytest.mark.asyncio
    async def test_get_boost_guidance_other_response(self, boost_manager):
        """Test get_boost_guidance with OTHER response."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "This is just a regular response without proper formatting."
                }
            }]
        }

        boost_manager.client.post = AsyncMock(return_value=mock_response)

        response_type, analysis, guidance = await boost_manager.get_boost_guidance(
            "Test request", []
        )

        assert response_type == "OTHER"
        assert analysis == ""
        assert guidance == ""

    @pytest.mark.asyncio
    async def test_get_boost_guidance_analysis_only(self, boost_manager):
        """Test get_boost_guidance with only ANALYSIS section."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "ANALYSIS:\nThe user wants to know something but no tools are needed."
                }
            }]
        }

        boost_manager.client.post = AsyncMock(return_value=mock_response)

        response_type, analysis, guidance = await boost_manager.get_boost_guidance(
            "Simple question", []
        )

        assert response_type == "OTHER"
        assert analysis == "The user wants to know something but no tools are needed."
        assert guidance == ""

    @pytest.mark.asyncio
    async def test_close(self, boost_manager):
        """Test closing the HTTP client."""
        boost_manager.client.aclose = AsyncMock()
        await boost_manager.close()
        boost_manager.client.aclose.assert_called_once()