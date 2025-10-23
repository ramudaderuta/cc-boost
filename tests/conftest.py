import os

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Ensure default environment variables are present before config is imported.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("BOOST_BASE_URL", "https://boost.example.com/v1")
os.environ.setdefault("BOOST_API_KEY", "sk-boost-test-key")
os.environ.setdefault("ENABLE_BOOST_SUPPORT", "NONE")
os.environ.setdefault("ANTHROPIC_API_KEY", "")

from src.main import app  # noqa: E402


@pytest_asyncio.fixture
async def test_client():
    """Provide an AsyncClient wired to the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
