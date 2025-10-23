"""Unit tests for AuxiliaryModelBuilder."""

import pytest
from unittest.mock import MagicMock
from src.core.auxiliary_builder import AuxiliaryModelBuilder
from src.models.claude import ClaudeMessagesRequest


class TestAuxiliaryModelBuilder:
    """Test AuxiliaryModelBuilder functionality."""

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

    def test_build_auxiliary_request_from_claude_request(self, sample_tools):
        """Test building auxiliary request from Claude request object."""
        original_request = MagicMock()
        original_request.model = "claude-3-sonnet-20241022"
        original_request.messages = [
            {"role": "user", "content": "Read and analyze /tmp/test.txt"}
        ]
        original_request.stream = False
        original_request.max_tokens = 1000
        original_request.temperature = 0.7

        analysis = "The user wants to read and analyze a file."
        guidance = "1. Call read_file with path: '/tmp/test.txt'\n2. Analyze the content"

        result = AuxiliaryModelBuilder.build_auxiliary_request(
            original_request, analysis, guidance, sample_tools
        )

        # Check basic structure
        assert result["model"] == "claude-3-sonnet-20241022"
        assert result["stream"] is False
        assert result["max_tokens"] == 1000
        assert result["temperature"] == 0.7
        assert result["tools"] == sample_tools
        assert result["tool_choice"] == "auto"

        # Check messages structure
        assert len(result["messages"]) == 2  # system + user
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][1]["role"] == "user"

        # Check system message content
        system_content = result["messages"][0]["content"]
        assert "ANALYSIS:" in system_content
        assert "GUIDANCE:" in system_content
        assert analysis in system_content
        assert guidance in system_content
        assert "Please follow the guidance to complete the user's request" in system_content

    def test_build_auxiliary_request_from_dict(self, sample_tools):
        """Test building auxiliary request from dict request."""
        original_request = {
            "model": "claude-3-haiku-20241022",
            "messages": [
                {"role": "user", "content": "Simple request"}
            ],
            "stream": True,
            "max_tokens": 500
            # No temperature
        }

        analysis = "Simple analysis"
        guidance = "1. Call simple_tool"

        result = AuxiliaryModelBuilder.build_auxiliary_request(
            original_request, analysis, guidance, sample_tools
        )

        assert result["model"] == "claude-3-haiku-20241022"
        assert result["stream"] is True
        assert result["max_tokens"] == 500
        assert "temperature" not in result  # Should not be included
        assert result["tools"] == sample_tools

    def test_build_auxiliary_request_with_existing_system_message(self, sample_tools):
        """Test building auxiliary request when original has system message."""
        original_request = MagicMock()
        original_request.model = "claude-3-sonnet-20241022"
        original_request.messages = [
            {"role": "system", "content": "Original system message"},
            {"role": "user", "content": "User request"}
        ]
        original_request.stream = False
        original_request.max_tokens = None
        original_request.temperature = None

        analysis = "Analysis content"
        guidance = "Guidance content"

        result = AuxiliaryModelBuilder.build_auxiliary_request(
            original_request, analysis, guidance, sample_tools
        )

        # Should have 2 messages: new system + user (original system excluded)
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][1]["content"] == "User request"

        # Original system message should be replaced
        assert "Original system message" not in result["messages"][0]["content"]

    def test_build_auxiliary_request_without_optional_params(self, sample_tools):
        """Test building auxiliary request without optional parameters."""
        original_request = MagicMock()
        original_request.model = "claude-3-sonnet-20241022"
        original_request.messages = [{"role": "user", "content": "Test"}]
        # Configure MagicMock to return None for missing attributes
        original_request.max_tokens = None
        original_request.temperature = None
        original_request.stream = False  # This has a default

        analysis = "Test analysis"
        guidance = "Test guidance"

        result = AuxiliaryModelBuilder.build_auxiliary_request(
            original_request, analysis, guidance, sample_tools
        )

        # Should only have required parameters
        assert "model" in result
        assert "messages" in result
        assert "stream" in result
        assert "tools" in result
        assert "tool_choice" in result
        assert "max_tokens" not in result
        assert "temperature" not in result

    def test_build_auxiliary_request_complex_analysis_and_guidance(self, sample_tools):
        """Test building auxiliary request with complex analysis and guidance."""
        original_request = MagicMock()
        original_request.model = "claude-3-sonnet-20241022"
        original_request.messages = [{"role": "user", "content": "Complex task"}]
        original_request.stream = False

        analysis = """The user wants to process a complex data pipeline.
Initial context: We have CSV files in /data/ directory.
First uncertainty: File format and structure.
Potential path: Read files, validate structure, process data.
Refining thought: Start with file discovery, then validation."""

        guidance = """1. Call bash with command: 'find /data -name "*.csv" -type f'
2. Call read_file with path: '/data/schema.json'
3. Call bash with command: 'python process_pipeline.py /data /output'
4. Call write_file with path: '/output/summary.txt' and content: 'Processing complete'"""

        result = AuxiliaryModelBuilder.build_auxiliary_request(
            original_request, analysis, guidance, sample_tools
        )

        system_content = result["messages"][0]["content"]
        assert "data pipeline" in system_content
        assert "find /data -name \"*.csv\" -type f" in system_content
        assert "python process_pipeline.py /data /output" in system_content

    def test_build_auxiliary_request_empty_analysis_and_guidance(self, sample_tools):
        """Test building auxiliary request with empty analysis and guidance."""
        original_request = MagicMock()
        original_request.model = "claude-3-sonnet-20241022"
        original_request.messages = [{"role": "user", "content": "Simple task"}]

        analysis = ""
        guidance = ""

        result = AuxiliaryModelBuilder.build_auxiliary_request(
            original_request, analysis, guidance, sample_tools
        )

        system_content = result["messages"][0]["content"]
        assert "ANALYSIS:" in system_content
        assert "GUIDANCE:" in system_content
        # Should still create valid structure even with empty content

    def test_detect_tool_usage_with_tool_calls(self):
        """Test tool usage detection when tools are called."""
        response = {
            "choices": [{
                "message": {
                    "content": "I'll help you with that.",
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

        assert AuxiliaryModelBuilder.detect_tool_usage(response) is True

    def test_detect_tool_usage_without_tool_calls(self):
        """Test tool usage detection when no tools are called."""
        response = {
            "choices": [{
                "message": {
                    "content": "I can help you without using any tools."
                }
            }]
        }

        assert AuxiliaryModelBuilder.detect_tool_usage(response) is False

    def test_detect_tool_usage_empty_tool_calls(self):
        """Test tool usage detection with empty tool calls."""
        response = {
            "choices": [{
                "message": {
                    "content": "No tools used.",
                    "tool_calls": []
                }
            }]
        }

        assert AuxiliaryModelBuilder.detect_tool_usage(response) is False

    def test_detect_tool_usage_streaming_response_with_tools(self):
        """Test tool usage detection in streaming response."""
        response = {
            "delta": {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": "{\"command\": \"ls\"}"
                        }
                    }
                ]
            }
        }

        assert AuxiliaryModelBuilder.detect_tool_usage(response) is True

    def test_detect_tool_usage_streaming_response_without_tools(self):
        """Test tool usage detection in streaming response without tools."""
        response = {
            "delta": {
                "content": "This is a streaming response without tools."
            }
        }

        assert AuxiliaryModelBuilder.detect_tool_usage(response) is False

    def test_detect_tool_usage_empty_response(self):
        """Test tool usage detection with empty response."""
        response = {}

        assert AuxiliaryModelBuilder.detect_tool_usage(response) is False

    def test_detect_tool_usage_no_choices(self):
        """Test tool usage detection when no choices are present."""
        response = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }

        assert AuxiliaryModelBuilder.detect_tool_usage(response) is False

    def test_extract_final_response_with_content(self):
        """Test extracting final response when content is present."""
        response = {
            "choices": [{
                "message": {
                    "content": "This is the final response content."
                }
            }]
        }

        result = AuxiliaryModelBuilder.extract_final_response(response)
        assert result == "This is the final response content."

    def test_extract_final_response_empty_content(self):
        """Test extracting final response when content is empty."""
        response = {
            "choices": [{
                "message": {
                    "content": ""
                }
            }]
        }

        result = AuxiliaryModelBuilder.extract_final_response(response)
        assert result == ""

    def test_extract_final_response_no_content_with_tool_calls(self):
        """Test extracting final response when no content but tool calls exist."""
        response = {
            "choices": [{
                "message": {
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

        result = AuxiliaryModelBuilder.extract_final_response(response)
        assert result == ""

    def test_extract_final_response_no_message(self):
        """Test extracting final response when no message is present."""
        response = {
            "choices": [{}]
        }

        result = AuxiliaryModelBuilder.extract_final_response(response)
        assert result == ""

    def test_extract_final_response_no_choices(self):
        """Test extracting final response when no choices are present."""
        response = {}

        result = AuxiliaryModelBuilder.extract_final_response(response)
        assert result == ""

    def test_extract_final_response_none_content(self):
        """Test extracting final response when content is None."""
        response = {
            "choices": [{
                "message": {
                    "content": None
                }
            }]
        }

        result = AuxiliaryModelBuilder.extract_final_response(response)
        assert result == ""