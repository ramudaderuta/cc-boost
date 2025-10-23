"""Unit tests for boost configuration system."""

import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock
from src.core.config import Config


class TestBoostConfiguration:
    """Test boost configuration loading and validation."""

    def test_boost_configuration_none_by_default(self):
        """Test that boost is disabled by default."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'sk-test-key',
                'ANTHROPIC_API_KEY': 'ant-test-key'
            }):
                config = Config()
                assert config.enable_boost_support == "NONE"
                assert not config.is_boost_enabled_for_model("BIG_MODEL")
                assert not config.is_boost_enabled_for_model("MIDDLE_MODEL")
                assert not config.is_boost_enabled_for_model("SMALL_MODEL")

    def test_boost_configuration_big_model_enabled(self):
        """Test boost enabled for BIG_MODEL."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'sk-test-key',
                'ANTHROPIC_API_KEY': 'ant-test-key',
                'ENABLE_BOOST_SUPPORT': 'BIG_MODEL',
                'BOOST_BASE_URL': 'https://api.test.com/v1',
                'BOOST_API_KEY': 'sk-boost-test-key'
            }):
                config = Config()
                assert config.enable_boost_support == "BIG_MODEL"
                assert config.is_boost_enabled_for_model("BIG_MODEL")
                assert not config.is_boost_enabled_for_model("MIDDLE_MODEL")
                assert not config.is_boost_enabled_for_model("SMALL_MODEL")

    def test_boost_configuration_middle_model_enabled(self):
        """Test boost enabled for MIDDLE_MODEL."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'sk-test-key',
                'ANTHROPIC_API_KEY': 'ant-test-key',
                'ENABLE_BOOST_SUPPORT': 'MIDDLE_MODEL',
                'BOOST_BASE_URL': 'https://api.test.com/v1',
                'BOOST_API_KEY': 'sk-boost-test-key'
            }):
                config = Config()
                assert config.enable_boost_support == "MIDDLE_MODEL"
                assert not config.is_boost_enabled_for_model("BIG_MODEL")
                assert config.is_boost_enabled_for_model("MIDDLE_MODEL")
                assert not config.is_boost_enabled_for_model("SMALL_MODEL")

    def test_boost_configuration_small_model_enabled(self):
        """Test boost enabled for SMALL_MODEL."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'sk-test-key',
                'ANTHROPIC_API_KEY': 'ant-test-key',
                'ENABLE_BOOST_SUPPORT': 'SMALL_MODEL',
                'BOOST_BASE_URL': 'https://api.test.com/v1',
                'BOOST_API_KEY': 'sk-boost-test-key'
            }):
                config = Config()
                assert config.enable_boost_support == "SMALL_MODEL"
                assert not config.is_boost_enabled_for_model("BIG_MODEL")
                assert not config.is_boost_enabled_for_model("MIDDLE_MODEL")
                assert config.is_boost_enabled_for_model("SMALL_MODEL")

    def test_boost_configuration_invalid_boost_support_value(self):
        """Test error when invalid ENABLE_BOOST_SUPPORT value is provided."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'sk-test-key',
                'ANTHROPIC_API_KEY': 'ant-test-key',
                'ENABLE_BOOST_SUPPORT': 'INVALID_VALUE'
            }):
                with pytest.raises(ValueError, match="ENABLE_BOOST_SUPPORT must be one of"):
                    Config()

    def test_boost_configuration_missing_base_url(self):
        """Test error when BOOST_BASE_URL is missing but boost is enabled."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'sk-test-key',
                'ANTHROPIC_API_KEY': 'ant-test-key',
                'ENABLE_BOOST_SUPPORT': 'MIDDLE_MODEL',
                'BOOST_API_KEY': 'sk-boost-test-key'
                # Missing BOOST_BASE_URL
            }):
                with pytest.raises(ValueError, match="BOOST_BASE_URL is required"):
                    Config()

    def test_boost_configuration_missing_api_key(self):
        """Test error when BOOST_API_KEY is missing but boost is enabled."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'sk-test-key',
                'ANTHROPIC_API_KEY': 'ant-test-key',
                'ENABLE_BOOST_SUPPORT': 'MIDDLE_MODEL',
                'BOOST_BASE_URL': 'https://api.test.com/v1'
                # Missing BOOST_API_KEY
            }):
                with pytest.raises(ValueError, match="BOOST_API_KEY is required"):
                    Config()

    def test_boost_configuration_default_values(self):
        """Test default boost configuration values."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'sk-test-key',
                'ANTHROPIC_API_KEY': 'ant-test-key',
                'ENABLE_BOOST_SUPPORT': 'MIDDLE_MODEL',
                'BOOST_BASE_URL': 'https://api.test.com/v1',
                'BOOST_API_KEY': 'sk-boost-test-key'
            }):
                config = Config()
                assert config.boost_model == "gpt-4o"  # Default value
                assert config.boost_wrapper_template is None  # Default value

    def test_boost_configuration_custom_values(self):
        """Test custom boost configuration values."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'sk-test-key',
                'ANTHROPIC_API_KEY': 'ant-test-key',
                'ENABLE_BOOST_SUPPORT': 'BIG_MODEL',
                'BOOST_BASE_URL': 'https://custom-api.com/v1',
                'BOOST_API_KEY': 'sk-custom-boost-key',
                'BOOST_MODEL': 'gpt-5-preview',
                'BOOST_WRAPPER_TEMPLATE': 'Custom template with {loop_count} and {user_request}'
            }):
                config = Config()
                assert config.boost_model == "gpt-5-preview"
                assert config.boost_wrapper_template == 'Custom template with {loop_count} and {user_request}'

    def test_boost_configuration_case_insensitive(self):
        """Test that ENABLE_BOOST_SUPPORT is case insensitive."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'sk-test-key',
                'ANTHROPIC_API_KEY': 'ant-test-key',
                'ENABLE_BOOST_SUPPORT': 'middle_model',  # lowercase
                'BOOST_BASE_URL': 'https://api.test.com/v1',
                'BOOST_API_KEY': 'sk-boost-test-key'
            }):
                config = Config()
                assert config.enable_boost_support == "MIDDLE_MODEL"
                assert config.is_boost_enabled_for_model("MIDDLE_MODEL")

    def test_boost_configuration_no_validation_when_disabled(self):
        """Test that boost configuration is not validated when boost is disabled."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'sk-test-key',
                'ANTHROPIC_API_KEY': 'ant-test-key',
                'ENABLE_BOOST_SUPPORT': 'NONE'
                # No boost config needed
            }):
                # Should not raise any errors
                config = Config()
                assert config.enable_boost_support == "NONE"
                assert config.boost_base_url is None
                assert config.boost_api_key is None