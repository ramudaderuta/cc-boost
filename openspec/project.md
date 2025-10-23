# Project Context

## Purpose
Claude Code Proxy is a bridge server that enables Claude Code CLI to work with any OpenAI-compatible API provider. It converts Claude API requests to OpenAI format, allowing users to use various LLM providers (OpenAI, Azure OpenAI, local models via Ollama, etc.) through the familiar Claude Code interface while maintaining full compatibility with Claude's API features including function calling, streaming, image support, and tool use.

## Tech Stack
- **Language**: Python 3.9+ (supports up to 3.12)
- **Web Framework**: FastAPI 0.115.11+ with async support
- **ASGI Server**: Uvicorn 0.34.0+
- **API Client**: OpenAI Python SDK 1.54.0+
- **Data Validation**: Pydantic 2.0.0+
- **Environment Management**: python-dotenv 1.0.0+
- **Package Manager**: UV (recommended) with pip fallback
- **Containerization**: Docker with docker-compose support

## Project Conventions

### Code Style
- **Formatter**: Black with 100-character line length
- **Import Sorting**: isort with Black profile
- **Type Checking**: mypy with strict settings (disallow_untyped_defs)
- **Python Version**: Target Python 3.8+ for compatibility
- **Code Organization**: Modular structure under src/ directory

### Architecture Patterns
- **Pattern**: Clean architecture with separation of concerns
  - `src/api/` - API endpoints and routing
  - `src/core/` - Configuration, logging, model management
  - `src/models/` - Data models for Claude and OpenAI APIs
  - `src/conversion/` - Request/response converters between API formats
- **Async/Await**: Full async implementation for high concurrency
- **Dependency Injection**: FastAPI's dependency system for configuration and clients
- **Environment-based Config**: All settings via environment variables with .env support

### Testing Strategy
- **Test Framework**: pytest with pytest-asyncio for async tests
- **HTTP Testing**: httpx for mock API calls
- **Test Location**: Tests in `src/test_claude_to_openai.py`
- **Coverage**: Comprehensive testing of proxy functionality including:
  - Request/response conversion
  - Model mapping
  - Error handling
  - Streaming responses

### Git Workflow
- **Branching**: Main branch for stable releases
- **Commit Format**: Conventional commits with clear descriptions
- **Versioning**: Semantic versioning (currently 1.0.0)
- **License**: MIT License

## Domain Context
This is an API translation proxy that sits between Claude Code CLI and LLM providers. Key domain concepts:
- **Model Mapping**: Claude models (haiku/sonnet/opus) mapped to configurable target models
- **API Translation**: Converts Claude's `/v1/messages` format to OpenAI's chat completions format
- **Function Calling**: Translates Claude's tool_use format to OpenAI's function calling
- **Streaming**: Maintains SSE streaming compatibility for real-time responses
- **Authentication**: Optional API key validation for client access control

## Important Constraints
- **API Compatibility**: Must maintain full compatibility with Claude Code's expected API surface
- **Performance**: Low latency overhead for proxy operations (< 50ms added latency)
- **Reliability**: Robust error handling and retry mechanisms (max 2 retries)
- **Security**: Optional API key validation, no credential storage
- **Resource Limits**: Configurable token limits (default 4096) and timeouts (90s)
- **Concurrency**: Must handle multiple simultaneous requests efficiently

## External Dependencies
- **OpenAI-compatible APIs**: Primary dependency on target LLM provider APIs
- **Docker**: Optional containerized deployment
- **Environment Variables**: Configuration via .env file or shell environment
- **Python Package Index**: Dependencies hosted on PyPI
- **GitHub Repository**: Source code and issue tracking at https://github.com/holegots/cc-boost
