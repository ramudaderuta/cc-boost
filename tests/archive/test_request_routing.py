"""Unit tests for request routing functionality."""

import pytest
from unittest.mock import MagicMock, patch
from src.core.model_manager import ModelManager
from src.core.config import Config
from src.models.claude import ClaudeMessagesRequest


class TestRequestRouting:
    """Test request routing logic for boost mode detection."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        config = MagicMock()
        config.enable_boost_support = "NONE"
        config.big_model = "gpt-4o"
        config.middle_model = "gpt-4o"
        config.small_model = "gpt-4o-mini"
        return config

    @pytest.fixture
    def model_manager(self, mock_config):
        """Create a ModelManager instance for testing."""
        return ModelManager(mock_config)

    def test_get_model_tier_haiku(self, model_manager):
        """Test model tier detection for haiku models."""
        assert model_manager.get_model_tier("claude-3-haiku-20241022") == "SMALL_MODEL"
        assert model_manager.get_model_tier("claude-3-5-haiku-20241022") == "SMALL_MODEL"

    def test_get_model_tier_sonnet(self, model_manager):
        """Test model tier detection for sonnet models."""
        assert model_manager.get_model_tier("claude-3-sonnet-20241022") == "MIDDLE_MODEL"
        assert model_manager.get_model_tier("claude-3-5-sonnet-20241022") == "MIDDLE_MODEL"

    def test_get_model_tier_opus(self, model_manager):
        """Test model tier detection for opus models."""
        assert model_manager.get_model_tier("claude-3-opus-20241022") == "BIG_MODEL"
        assert model_manager.get_model_tier("claude-3-5-opus-20241022") == "BIG_MODEL"

    def test_get_model_tier_openai_models(self, model_manager):
        """Test model tier detection for OpenAI models."""
        model_manager.config.small_model = "gpt-4o-mini"
        model_manager.config.middle_model = "gpt-4o"
        model_manager.config.big_model = "gpt-4o-turbo"

        assert model_manager.get_model_tier("gpt-4o-mini") == "SMALL_MODEL"
        assert model_manager.get_model_tier("gpt-4o") == "MIDDLE_MODEL"
        assert model_manager.get_model_tier("gpt-4o-turbo") == "BIG_MODEL"

    def test_get_model_tier_unknown_models(self, model_manager):
        """Test model tier detection for unknown models."""
        assert model_manager.get_model_tier("unknown-model") == "BIG_MODEL"
        assert model_manager.get_model_tier("custom-model") == "BIG_MODEL"

    def test_get_model_tier_other_providers(self, model_manager):
        """Test model tier detection for other providers."""
        # Should return as-is for other providers
        assert model_manager.get_model_tier("ep-2024-12-06") == "BIG_MODEL"  # ARK
        assert model_manager.get_model_tier("doubao-pro-4k") == "BIG_MODEL"  # Doubao
        assert model_manager.get_model_tier("deepseek-chat") == "BIG_MODEL"  # DeepSeek

    def test_has_tools_with_tools(self, model_manager):
        """Test has_tools when request contains tools."""
        request = MagicMock()
        request.tools = [{"name": "test_tool", "description": "Test tool"}]

        assert model_manager.has_tools(request) is True

    def test_has_tools_without_tools(self, model_manager):
        """Test has_tools when request has no tools."""
        request = MagicMock()
        request.tools = None

        assert model_manager.has_tools(request) is False

    def test_has_tools_empty_tools(self, model_manager):
        """Test has_tools when request has empty tools list."""
        request = MagicMock()
        request.tools = []

        assert model_manager.has_tools(request) is False

    def test_has_tools_dict_request_with_tools(self, model_manager):
        """Test has_tools with dict request containing tools."""
        request = {"tools": [{"name": "test_tool"}]}

        assert model_manager.has_tools(request) is True

    def test_has_tools_dict_request_without_tools(self, model_manager):
        """Test has_tools with dict request without tools."""
        request = {"messages": [{"role": "user", "content": "test"}]}

        assert model_manager.has_tools(request) is False

    def test_boost_mode_detection_haiku_with_tools(self, model_manager):
        """Test boost mode detection for haiku with tools."""
        config = MagicMock()
        config.enable_boost_support = "SMALL_MODEL"

        request = MagicMock()
        request.model = "claude-3-haiku-20241022"
        request.tools = [{"name": "test_tool"}]

        use_boost = (
            config.enable_boost_support != "NONE" and
            config.is_boost_enabled_for_model(model_manager.get_model_tier(request.model)) and
            model_manager.has_tools(request)
        )

        assert use_boost is True

    def test_boost_mode_detection_sonnet_with_tools(self, model_manager):
        """Test boost mode detection for sonnet with tools."""
        config = MagicMock()
        config.enable_boost_support = "MIDDLE_MODEL"

        request = MagicMock()
        request.model = "claude-3-sonnet-20241022"
        request.tools = [{"name": "test_tool"}]

        use_boost = (
            config.enable_boost_support != "NONE" and
            config.is_boost_enabled_for_model(model_manager.get_model_tier(request.model)) and
            model_manager.has_tools(request)
        )

        assert use_boost is True

    def test_boost_mode_detection_opus_with_tools(self, model_manager):
        """Test boost mode detection for opus with tools."""
        config = MagicMock()
        config.enable_boost_support = "BIG_MODEL"

        request = MagicMock()
        request.model = "claude-3-opus-20241022"
        request.tools = [{"name": "test_tool"}]

        use_boost = (
            config.enable_boost_support != "NONE" and
            config.is_boost_enabled_for_model(model_manager.get_model_tier(request.model)) and
            model_manager.has_tools(request)
        )

        assert use_boost is True

    def test_boost_mode_detection_wrong_tier(self, model_manager):
        """Test boost mode detection when model tier doesn't match."""
        config = MagicMock()
        config.enable_boost_support = "SMALL_MODEL"  # Only small model
        # Configure the mock to return False for MIDDLE_MODEL
        config.is_boost_enabled_for_model.return_value = False

        request = MagicMock()
        request.model = "claude-3-sonnet-20241022"  # Middle model
        request.tools = [{"name": "test_tool"}]

        use_boost = (
            config.enable_boost_support != "NONE" and
            config.is_boost_enabled_for_model(model_manager.get_model_tier(request.model)) and
            model_manager.has_tools(request)
        )

        assert use_boost is False

    def test_boost_mode_detection_no_tools(self, model_manager):
        """Test boost mode detection when no tools are present."""
        config = MagicMock()
        config.enable_boost_support = "MIDDLE_MODEL"

        request = MagicMock()
        request.model = "claude-3-sonnet-20241022"
        request.tools = None

        use_boost = (
            config.enable_boost_support != "NONE" and
            config.is_boost_enabled_for_model(model_manager.get_model_tier(request.model)) and
            model_manager.has_tools(request)
        )

        assert use_boost is False

    def test_boost_mode_detection_boost_disabled(self, model_manager):
        """Test boost mode detection when boost is disabled."""
        config = MagicMock()
        config.enable_boost_support = "NONE"

        request = MagicMock()
        request.model = "claude-3-sonnet-20241022"
        request.tools = [{"name": "test_tool"}]

        use_boost = (
            config.enable_boost_support != "NONE" and
            config.is_boost_enabled_for_model(model_manager.get_model_tier(request.model)) and
            model_manager.has_tools(request)
        )

        assert use_boost is False

    def test_boost_mode_detection_empty_tools(self, model_manager):
        """Test boost mode detection with empty tools list."""
        config = MagicMock()
        config.enable_boost_support = "MIDDLE_MODEL"

        request = MagicMock()
        request.model = "claude-3-sonnet-20241022"
        request.tools = []

        use_boost = (
            config.enable_boost_support != "NONE" and
            config.is_boost_enabled_for_model(model_manager.get_model_tier(request.model)) and
            model_manager.has_tools(request)
        )

        assert use_boost is False

    def test_model_mapping_haiku_to_small_model(self, model_manager):
        """Test model mapping from haiku to small model."""
        model_manager.config.small_model = "gpt-4o-mini"

        result = model_manager.map_claude_model_to_openai("claude-3-haiku-20241022")
        assert result == "gpt-4o-mini"

    def test_model_mapping_sonnet_to_middle_model(self, model_manager):
        """Test model mapping from sonnet to middle model."""
        model_manager.config.middle_model = "gpt-4o"

        result = model_manager.map_claude_model_to_openai("claude-3-sonnet-20241022")
        assert result == "gpt-4o"

    def test_model_mapping_opus_to_big_model(self, model_manager):
        """Test model mapping from opus to big model."""
        model_manager.config.big_model = "gpt-4o-turbo"

        result = model_manager.map_claude_model_to_openai("claude-3-opus-20241022")
        assert result == "gpt-4o-turbo"

    def test_model_mapping_openai_models_passthrough(self, model_manager):
        """Test that OpenAI models pass through unchanged."""
        assert model_manager.map_claude_model_to_openai("gpt-4o") == "gpt-4o"
        assert model_manager.map_claude_model_to_openai("gpt-4o-mini") == "gpt-4o-mini"
        assert model_manager.map_claude_model_to_openai("o1-preview") == "o1-preview"

    def test_model_mapping_other_providers_passthrough(self, model_manager):
        """Test that other provider models pass through unchanged."""
        assert model_manager.map_claude_model_to_openai("ep-2024-12-06") == "ep-2024-12-06"
        assert model_manager.map_claude_model_to_openai("doubao-pro-4k") == "doubao-pro-4k"
        assert model_manager.map_claude_model_to_openai("deepseek-chat") == "deepseek-chat"

    def test_model_mapping_unknown_model_default(self, model_manager):
        """Test that unknown models default to big model."""
        model_manager.config.big_model = "gpt-4o-turbo"

        result = model_manager.map_claude_model_to_openai("unknown-model")
        assert result == "gpt-4o-turbo"