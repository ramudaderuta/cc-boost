"""Tests for multi-provider compatibility in boost mode."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.boost_model_manager import BoostModelManager
from src.core.boost_orchestrator import BoostOrchestrator
from src.core.config import Config
from src.models.claude import ClaudeMessagesRequest


class TestMultiProviderCompatibility:
    """Test compatibility with different model providers."""

    def _sample_tools(self):
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
                },
            {
                    "name": "bash",
                    "description": "Execute bash commands",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Command to execute"
                            }
                        },
                        "required": ["command"]
                    }
                }
            ]

    def test_boost_manager_openai_compatibility(self):
        """Test BoostModelManager with OpenAI provider."""
        config = MagicMock()
        config.boost_base_url = "https://api.openai.com/v1"
        config.boost_api_key = "sk-openai-key"
        config.boost_model = "gpt-4o"
        config.request_timeout = 30

        manager = BoostModelManager(config)

        # Test message building
        message = manager.build_boost_message(
            "Test request", self._sample_tools(), loop_count=1
        )

        assert message["model"] == "gpt-4o"
        assert "tools parameter" not in message
        assert "read_file: Read a file from filesystem" in message["messages"][0]["content"]

    def test_boost_manager_anthropic_compatibility(self):
        """Test BoostModelManager with Anthropic provider."""
        config = MagicMock()
        config.boost_base_url = "https://api.anthropic.com/v1"
        config.boost_api_key = "sk-ant-key"
        config.boost_model = "claude-3-5-sonnet-20241022"
        config.request_timeout = 60

        manager = BoostModelManager(config)

        # Test message building
        message = manager.build_boost_message(
            "Test request", self._sample_tools(), loop_count=0
        )

        assert message["model"] == "claude-3-5-sonnet-20241022"
        assert "tools parameter" not in message

    def test_boost_manager_azure_compatibility(self):
        """Test BoostModelManager with Azure OpenAI provider."""
        config = MagicMock()
        config.boost_base_url = "https://my-resource.openai.azure.com/openai/deployments/my-deployment"
        config.boost_api_key = "sk-azure-key"
        config.boost_model = "gpt-4o"
        config.request_timeout = 90

        manager = BoostModelManager(config)

        # Test message building
        message = manager.build_boost_message(
            "Test request", self._sample_tools(), loop_count=2
        )

        assert message["model"] == "gpt-4o"
        assert "tools parameter" not in message
        assert "Current ReAct Loop: 2" in message["messages"][0]["content"]

    def test_boost_manager_google_compatibility(self):
        """Test BoostModelManager with Google AI provider."""
        config = MagicMock()
        config.boost_base_url = "https://generativelanguage.googleapis.com/v1beta"
        config.boost_api_key = "google-api-key"
        config.boost_model = "gemini-1.5-pro"
        config.request_timeout = 30

        manager = BoostModelManager(config)

        # Test message building
        message = manager.build_boost_message(
            "Test request", self._sample_tools(), loop_count=1
        )

        assert message["model"] == "gemini-1.5-pro"
        assert "tools parameter" not in message

    def test_boost_manager_cohere_compatibility(self):
        """Test BoostModelManager with Cohere provider."""
        config = MagicMock()
        config.boost_base_url = "https://api.cohere.com/v1"
        config.boost_api_key = "cohere-api-key"
        config.boost_model = "command-r-plus"
        config.request_timeout = 45

        manager = BoostModelManager(config)

        # Test message building
        message = manager.build_boost_message(
            "Test request", self._sample_tools(), loop_count=0
        )

        assert message["model"] == "command-r-plus"
        assert "tools parameter" not in message

    def test_boost_manager_mistral_compatibility(self):
        """Test BoostModelManager with Mistral provider."""
        config = MagicMock()
        config.boost_base_url = "https://api.mistral.ai/v1"
        config.boost_api_key = "mistral-api-key"
        config.boost_model = "mistral-large"
        config.request_timeout = 30

        manager = BoostModelManager(config)

        # Test message building
        message = manager.build_boost_message(
            "Test request", self._sample_tools(), loop_count=1
        )

        assert message["model"] == "mistral-large"
        assert "tools parameter" not in message

    def test_boost_manager_perplexity_compatibility(self):
        """Test BoostModelManager with Perplexity provider."""
        config = MagicMock()
        config.boost_base_url = "https://api.perplexity.ai"
        config.boost_api_key = "ppl-api-key"
        config.boost_model = "llama-3.1-sonar-large-128k-online"
        config.request_timeout = 60

        manager = BoostModelManager(config)

        # Test message building
        message = manager.build_boost_message(
            "Test request", self._sample_tools(), loop_count=0
        )

        assert message["model"] == "llama-3.1-sonar-large-128k-online"
        assert "tools parameter" not in message

    def test_boost_manager_groq_compatibility(self):
        """Test BoostModelManager with Groq provider."""
        config = MagicMock()
        config.boost_base_url = "https://api.groq.com/openai/v1"
        config.boost_api_key = "gsk-api-key"
        config.boost_model = "llama-3.1-70b-versatile"
        config.request_timeout = 30

        manager = BoostModelManager(config)

        # Test message building
        message = manager.build_boost_message(
            "Test request", self._sample_tools(), loop_count=1
        )

        assert message["model"] == "llama-3.1-70b-versatile"
        assert "tools parameter" not in message

    def test_boost_manager_together_compatibility(self):
        """Test BoostModelManager with Together AI provider."""
        config = MagicMock()
        config.boost_base_url = "https://api.together.xyz/v1"
        config.boost_api_key = "together-api-key"
        config.boost_model = "meta-llama/Llama-3-70b-chat-hf"
        config.request_timeout = 45

        manager = BoostModelManager(config)

        # Test message building
        message = manager.build_boost_message(
            "Test request", self._sample_tools(), loop_count=2
        )

        assert message["model"] == "meta-llama/Llama-3-70b-chat-hf"
        assert "tools parameter" not in message

    def test_boost_manager_fireworks_compatibility(self):
        """Test BoostModelManager with Fireworks AI provider."""
        config = MagicMock()
        config.boost_base_url = "https://api.fireworks.ai/inference/v1"
        config.boost_api_key = "fw-api-key"
        config.boost_model = "accounts/fireworks/models/llama-v3p-70b-instruct"
        config.request_timeout = 60

        manager = BoostModelManager(config)

        # Test message building
        message = manager.build_boost_message(
            "Test request", self._sample_tools(), loop_count=0
        )

        assert message["model"] == "accounts/fireworks/models/llama-v3p-70b-instruct"
        assert "tools parameter" not in message

    def test_boost_manager_anyscale_compatibility(self):
        """Test BoostModelManager with Anyscale provider."""
        config = MagicMock()
        config.boost_base_url = "https://api.endpoints.anyscale.com/v1"
        config.boost_api_key = "anyscale-api-key"
        config.boost_model = "meta-llama/Llama-2-70b-chat-hf"
        config.request_timeout = 90

        manager = BoostModelManager(config)

        # Test message building
        message = manager.build_boost_message(
            "Test request", self._sample_tools(), loop_count=1
        )

        assert message["model"] == "meta-llama/Llama-2-70b-chat-hf"
        assert "tools parameter" not in message

    def test_boost_manager_replicate_compatibility(self):
        """Test BoostModelManager with Replicate provider."""
        config = MagicMock()
        config.boost_base_url = "https://api.replicate.com/v1"
        config.boost_api_key = "replicate-api-key"
        config.boost_model = "meta/meta-llama-3-70b-instruct"
        config.request_timeout = 120

        manager = BoostModelManager(config)

        # Test message building
        message = manager.build_boost_message(
            "Test request", self._sample_tools(), loop_count=2
        )

        assert message["model"] == "meta/meta-llama-3-70b-instruct"
        assert "tools parameter" not in message

    def test_boost_manager_ollama_compatibility(self):
        """Test BoostModelManager with Ollama provider."""
        config = MagicMock()
        config.boost_base_url = "http://localhost:11434/v1"
        config.boost_api_key = "ollama-api-key"
        config.boost_model = "llama3:70b"
        config.request_timeout = 300

        manager = BoostModelManager(config)

        # Test message building
        message = manager.build_boost_message(
            "Test request", self._sample_tools(), loop_count=0
        )

        assert message["model"] == "llama3:70b"
        assert "tools parameter" not in message

    def test_boost_manager_vllm_compatibility(self):
        """Test BoostModelManager with vLLM provider."""
        config = MagicMock()
        config.boost_base_url = "http://localhost:8000/v1"
        config.boost_api_key = "vllm-api-key"
        config.boost_model = "meta-llama/Meta-Llama-3-70B-Instruct"
        config.request_timeout = 60

        manager = BoostModelManager(config)

        # Test message building
        message = manager.build_boost_message(
            "Test request", self._sample_tools(), loop_count=1
        )

        assert message["model"] == "meta-llama/Meta-Llama-3-70B-Instruct"
        assert "tools parameter" not in message

    def test_boost_manager_custom_template_compatibility(self):
        """Test BoostModelManager with custom template across providers."""
        config = MagicMock()
        config.boost_base_url = "https://api.custom-provider.com/v1"
        config.boost_api_key = "custom-api-key"
        config.boost_model = "custom-model"
        config.boost_wrapper_template = "Custom template for {user_request} with tools: {tools_text}"
        config.request_timeout = 30

        manager = BoostModelManager(config)

        # Test message building with custom template
        message = manager.build_boost_message(
            "Test request", self._sample_tools(), loop_count=0
        )

        assert message["model"] == "custom-model"
        assert "tools parameter" not in message
        content = message["messages"][0]["content"]
        assert "Custom template for Test request with tools:" in content

    @pytest.mark.asyncio
    async def test_boost_orchestrator_multi_provider_flow(self):
        """Test BoostOrchestrator with different providers."""
        # Test with OpenAI as boost provider
        config = MagicMock()
        config.boost_base_url = "https://api.openai.com/v1"
        config.boost_api_key = "sk-openai-key"
        config.boost_model = "gpt-4o"
        config.enable_boost_support = "MIDDLE_MODEL"
        config.request_timeout = 30

        openai_client = MagicMock()
        openai_client.create_chat_completion = AsyncMock()
        openai_client.create_chat_completion_stream = AsyncMock()

        orchestrator = BoostOrchestrator(config, openai_client)

        # Mock boost manager to return SUMMARY
        orchestrator.boost_manager.get_boost_guidance = AsyncMock(
            return_value=("SUMMARY", "", "Final answer from OpenAI boost")
        )

        claude_request = ClaudeMessagesRequest(
            model="claude-3-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": "Test request"}],
            tools=self._sample_tools()
        )

        response = await orchestrator.execute_with_boost(claude_request, "test-request-id")

        assert response.content[0].text == "Final answer from OpenAI boost"

    @pytest.mark.asyncio
    async def test_boost_manager_timeout_compatibility(self):
        """Test BoostModelManager with different timeout settings."""
        timeout_values = [10, 30, 60, 120, 300]

        for timeout in timeout_values:
            config = MagicMock()
            config.boost_base_url = "https://api.test.com/v1"
            config.boost_api_key = "sk-test-key"
            config.boost_model = "test-model"
            config.request_timeout = timeout

            manager = BoostModelManager(config)

            # Test that timeout is configured correctly across phases
            client_timeout = manager.client.timeout
            assert client_timeout.connect == timeout
            assert client_timeout.read == timeout

    def test_boost_manager_header_compatibility(self):
        """Test BoostManager with different authentication headers."""
        config = MagicMock()
        config.boost_base_url = "https://api.test.com/v1"
        config.boost_api_key = "sk-test-key"
        config.boost_model = "test-model"
        config.request_timeout = 30

        manager = BoostModelManager(config)

        # Test that headers are set correctly
        expected_headers = {
            "Authorization": "Bearer sk-test-key",
            "Content-Type": "application/json"
        }

        headers = manager.client.headers
        for key, value in expected_headers.items():
            assert headers.get(key) == value

    def test_boost_manager_url_compatibility(self):
        """Test BoostManager with different URL formats."""
        url_formats = [
            "https://api.test.com/v1",
            "https://api.test.com/v1/",
            "http://localhost:8000/v1",
            "https://custom-domain.com/api/v1",
            "https://api.test.com:8443/v1"
        ]

        for url in url_formats:
            config = MagicMock()
            config.boost_base_url = url
            config.boost_api_key = "sk-test-key"
            config.boost_model = "test-model"
            config.request_timeout = 30

            manager = BoostModelManager(config)

            # Test that base URL resolves to the same location (httpx normalizes trailing slash)
            assert str(manager.client.base_url).rstrip('/') == url.rstrip('/')

    def test_boost_manager_model_name_compatibility(self):
        """Test BoostManager with different model naming conventions."""
        model_names = [
            "gpt-4o",
            "claude-3-5-sonnet-20241022",
            "gemini-1.5-pro",
            "command-r-plus",
            "mistral-large",
            "llama-3.1-70b-versatile",
            "meta-llama/Llama-3-70b-chat-hf",
            "accounts/fireworks/models/llama-v3p-70b-instruct",
            "llama3:70b"
        ]

        for model in model_names:
            config = MagicMock()
            config.boost_base_url = "https://api.test.com/v1"
            config.boost_api_key = "sk-test-key"
            config.boost_model = model
            config.request_timeout = 30

            manager = BoostModelManager(config)

            # Test message building with different model names
            message = manager.build_boost_message(
                "Test request", self._sample_tools(), loop_count=0
            )

            assert message["model"] == model

    def test_boost_manager_tools_formatting_compatibility(self):
        """Test BoostManager tools formatting across different tool schemas."""
        tool_schemas = [
            # Simple tool
            {
                "name": "simple_tool",
                "description": "A simple tool",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Input parameter"
                        }
                    },
                    "required": ["input"]
                }
            },
            # Complex tool with nested properties
            {
                "name": "complex_tool",
                "description": "A complex tool with nested schema",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "config": {
                            "type": "object",
                            "properties": {
                                "setting1": {
                                    "type": "string",
                                    "description": "First setting"
                                },
                                "setting2": {
                                    "type": "integer",
                                    "description": "Second setting"
                                }
                            }
                        },
                        "options": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of options"
                        }
                    },
                    "required": ["config"]
                }
            },
            # Tool without required fields
            {
                "name": "optional_tool",
                "description": "Tool with all optional parameters",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "optional1": {
                            "type": "string",
                            "description": "Optional parameter 1"
                        },
                        "optional2": {
                            "type": "boolean",
                            "description": "Optional parameter 2"
                        }
                    }
                }
            }
        ]

        config = MagicMock()
        config.boost_base_url = "https://api.test.com/v1"
        config.boost_api_key = "sk-test-key"
        config.boost_model = "test-model"
        config.request_timeout = 30

        manager = BoostModelManager(config)

        for tool in tool_schemas:
            # Test that tools are formatted correctly regardless of schema complexity
            tools_text = manager._format_tools_for_message([tool])
            assert tool["name"] in tools_text
            assert tool["description"] in tools_text

    def test_boost_manager_loop_context_compatibility(self):
        """Test BoostManager loop context across different providers."""
        config = MagicMock()
        config.boost_base_url = "https://api.test.com/v1"
        config.boost_api_key = "sk-test-key"
        config.boost_model = "test-model"
        config.request_timeout = 30

        manager = BoostModelManager(config)

        # Test different loop counts
        for loop_count in [0, 1, 2, 3]:
            message = manager.build_boost_message(
                "Test request", self._sample_tools(), loop_count=loop_count
            )

            content = message["messages"][0]["content"]
            assert f"Current ReAct Loop: {loop_count}" in content

        # Test with previous attempts
        previous_attempts = ["First attempt failed", "Second attempt failed"]
        message = manager.build_boost_message(
            "Test request", self._sample_tools(), loop_count=1, previous_attempts=previous_attempts
        )

        content = message["messages"][0]["content"]
        assert "First attempt failed" in content
        assert "Second attempt failed" in content
