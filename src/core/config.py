import os
import sys

# Configuration
class Config:
    def __init__(self):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Add Anthropic API key for client validation
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            print("Warning: ANTHROPIC_API_KEY not set. Client API key validation will be disabled.")

        self.openai_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.azure_api_version = os.environ.get("AZURE_API_VERSION")  # For Azure OpenAI
        self.host = os.environ.get("HOST", "0.0.0.0")
        self.port = int(os.environ.get("PORT", "8082"))
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        self.max_tokens_limit = int(os.environ.get("MAX_TOKENS_LIMIT", "4096"))
        self.min_tokens_limit = int(os.environ.get("MIN_TOKENS_LIMIT", "100"))

        # Connection settings
        self.request_timeout = int(os.environ.get("REQUEST_TIMEOUT", "90"))
        self.max_retries = int(os.environ.get("MAX_RETRIES", "2"))

        # Model settings - BIG and SMALL models
        self.big_model = os.environ.get("BIG_MODEL", "gpt-4o")
        self.middle_model = os.environ.get("MIDDLE_MODEL", self.big_model)
        self.small_model = os.environ.get("SMALL_MODEL", "gpt-4o-mini")

        # Boost model configuration
        self.boost_base_url = os.environ.get("BOOST_BASE_URL")
        self.boost_api_key = os.environ.get("BOOST_API_KEY")
        self.boost_model = os.environ.get("BOOST_MODEL", "gpt-4o")

        # Boost support configuration - which tiers to enable for
        boost_support = os.environ.get("ENABLE_BOOST_SUPPORT", "NONE").upper()
        valid_boost_support = ["NONE", "BIG_MODEL", "MIDDLE_MODEL", "SMALL_MODEL"]
        if boost_support not in valid_boost_support:
            raise ValueError(f"ENABLE_BOOST_SUPPORT must be one of {valid_boost_support}, got '{boost_support}'")
        self.enable_boost_support = boost_support

        # Custom wrapper template (optional)
        self.boost_wrapper_template = os.environ.get("BOOST_WRAPPER_TEMPLATE")

        # Validate boost configuration if enabled
        if self.enable_boost_support != "NONE":
            if not self.boost_base_url:
                raise ValueError("BOOST_BASE_URL is required when ENABLE_BOOST_SUPPORT is not NONE")
            if not self.boost_api_key:
                raise ValueError("BOOST_API_KEY is required when ENABLE_BOOST_SUPPORT is not NONE")

    def is_boost_enabled_for_model(self, model_tier: str) -> bool:
        """Check if boost is enabled for a specific model tier"""
        return self.enable_boost_support == model_tier

    def validate_api_key(self):
        """Basic API key validation"""
        if not self.openai_api_key:
            return False
        # Basic format check for OpenAI API keys
        if not self.openai_api_key.startswith('sk-'):
            return False
        return True

    def validate_client_api_key(self, client_api_key):
        """Validate client's Anthropic API key"""
        # If no ANTHROPIC_API_KEY is set in environment, skip validation
        if not self.anthropic_api_key:
            return True

        # Check if the client's API key matches the expected value
        return client_api_key == self.anthropic_api_key

    def get_custom_headers(self):
        """Get custom headers from environment variables"""
        custom_headers = {}

        # Get all environment variables
        env_vars = dict(os.environ)

        # Find CUSTOM_HEADER_* environment variables
        for env_key, env_value in env_vars.items():
            if env_key.startswith('CUSTOM_HEADER_'):
                # Convert CUSTOM_HEADER_KEY to Header-Key
                # Remove 'CUSTOM_HEADER_' prefix and convert to header format
                header_name = env_key[14:]  # Remove 'CUSTOM_HEADER_' prefix

                if header_name:  # Make sure it's not empty
                    # Convert underscores to hyphens for HTTP header format
                    header_name = header_name.replace('_', '-')
                    custom_headers[header_name] = env_value

        return custom_headers

def print_startup_help():
    """Print helpful startup information including available configuration options"""
    help_text = """
Configuration Options:
  Required:
    OPENAI_API_KEY         API key for the model provider
    ANTHROPIC_API_KEY      API key for client validation (optional)

  Optional:
    OPENAI_BASE_URL        Base URL for the model provider (default: https://api.openai.com/v1)
    AZURE_API_VERSION      API version for Azure OpenAI
    HOST                   Server host (default: 0.0.0.0)
    PORT                   Server port (default: 8082)
    LOG_LEVEL              Logging level (default: INFO)
    MAX_TOKENS_LIMIT       Maximum tokens limit (default: 4096)
    MIN_TOKENS_LIMIT       Minimum tokens limit (default: 100)
    REQUEST_TIMEOUT        Request timeout in seconds (default: 90)
    MAX_RETRIES            Maximum retry attempts (default: 2)

  Model Configuration:
    BIG_MODEL              Model name for big tier (default: gpt-4o)
    MIDDLE_MODEL           Model name for middle tier (default: BIG_MODEL)
    SMALL_MODEL            Model name for small tier (default: gpt-4o-mini)

  Boost Configuration (for tool-calling with providers that disable tools):
    ENABLE_BOOST_SUPPORT   Enable boost for which tier (NONE|BIG_MODEL|MIDDLE_MODEL|SMALL_MODEL)
    BOOST_BASE_URL         API endpoint for boost model
    BOOST_API_KEY          API key for boost model provider
    BOOST_MODEL            High-tier model for planning (default: gpt-4o)
    BOOST_WRAPPER_TEMPLATE Custom template for boost model prompt (optional)

  Custom Headers:
    CUSTOM_HEADER_*        Add custom HTTP headers (e.g., CUSTOM_HEADER_X_Custom=Value)
"""
    print(help_text)

try:
    config = Config()

    # Print configuration status
    boost_status = f"ENABLED for {config.enable_boost_support}" if config.enable_boost_support != "NONE" else "DISABLED"
    print(f"✓ Configuration loaded:")
    print(f"  - API_KEY: {'*' * 20}...")
    print(f"  - BASE_URL: {config.openai_base_url}")
    print(f"  - Models: BIG={config.big_model}, MIDDLE={config.middle_model}, SMALL={config.small_model}")
    print(f"  - Boost Support: {boost_status}")
    if config.enable_boost_support != "NONE":
        print(f"  - Boost Model: {config.boost_model}")
        print(f"  - Boost URL: {config.boost_base_url}")

    # Print help if requested
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print_startup_help()

except Exception as e:
    print(f"✗ Configuration Error: {e}")
    print("\nUse --help for configuration options")
    sys.exit(1)