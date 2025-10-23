"""Cancellation behaviour tests for the proxy."""

import asyncio
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.conversion.response_converter import convert_openai_streaming_to_claude_with_cancellation
from src.core.client import OpenAIClient
from src.models.claude import ClaudeMessagesRequest


@pytest.mark.asyncio
async def test_streaming_cancellation_triggers_cancel_request():
    """When the client disconnects mid-stream the OpenAI request should be cancelled."""

    async def fake_stream():
        yield "data: {\"choices\": []}"
        yield "data: [DONE]"

    # The HTTP request reports a disconnect on the second iteration.
    http_request = MagicMock()
    http_request.is_disconnected = AsyncMock(side_effect=[False, True])

    openai_client = MagicMock()
    logger = MagicMock()

    claude_request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{"role": "user", "content": "Write a long story."}],
        stream=True,
    )

    events: List[str] = []
    async for event in convert_openai_streaming_to_claude_with_cancellation(
        fake_stream(),
        claude_request,
        logger,
        http_request,
        openai_client,
        "req-123",
    ):
        events.append(event)

    openai_client.cancel_request.assert_called_once_with("req-123")
    assert any("event: " in chunk for chunk in events)


@pytest.mark.asyncio
async def test_cancel_request_sets_active_event():
    """OpenAIClient.cancel_request should set the cancellation event and report success."""
    client = OpenAIClient.__new__(OpenAIClient)
    client.active_requests = {"req-99": asyncio.Event()}

    assert not client.active_requests["req-99"].is_set()
    assert client.cancel_request("req-99") is True
    assert client.active_requests["req-99"].is_set()
    assert client.cancel_request("missing") is False
