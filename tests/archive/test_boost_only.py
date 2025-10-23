"""Functional tests for the boost-enabled HTTP flow."""

from unittest.mock import AsyncMock, patch

import pytest

from src.core.config import config
from src.models.claude import (
    ClaudeContentBlockText,
    ClaudeContentBlockToolUse,
    ClaudeMessageResponse,
    ClaudeUsage,
)


def _boost_available(monkeypatch, tier: str) -> None:
    """Enable boost for the requested model tier during a test."""
    monkeypatch.setattr(config, "enable_boost_support", tier)
    monkeypatch.setattr(config, "boost_base_url", "https://boost.example.com/v1")
    monkeypatch.setattr(config, "boost_api_key", "sk-boost-test-key")
    monkeypatch.setattr(
        config,
        "is_boost_enabled_for_model",
        lambda requested_tier: requested_tier == tier,
    )


@pytest.mark.asyncio
async def test_boost_mode_summary_response(test_client, monkeypatch):
    """Ensure boost SUMMARY responses surface through the proxy."""
    _boost_available(monkeypatch, "SMALL_MODEL")

    boost_response = ClaudeMessageResponse(
        id="msg_summary",
        type="message",
        role="assistant",
        content=[ClaudeContentBlockText(type="text", text="The answer is 4.")],
        model="claude-3-5-haiku-20241022",
        stop_reason="end_turn",
        usage=ClaudeUsage(input_tokens=10, output_tokens=4),
    )

    with patch(
        "src.api.endpoints.BoostOrchestrator.execute_with_boost",
        new=AsyncMock(return_value=boost_response),
    ) as mock_execute:
        response = await test_client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "What is 2 + 2?"}],
                "tools": [
                    {
                        "name": "calculator",
                        "description": "Perform basic arithmetic calculations",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "Expression to evaluate",
                                }
                            },
                            "required": ["expression"],
                        },
                    }
                ],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["content"][0]["text"] == "The answer is 4."
    assert mock_execute.await_count == 1


@pytest.mark.asyncio
async def test_boost_mode_guidance_response(test_client, monkeypatch):
    """Ensure GUIDANCE-style responses propagate tool use blocks."""
    _boost_available(monkeypatch, "SMALL_MODEL")

    boost_response = ClaudeMessageResponse(
        id="msg_guidance",
        type="message",
        role="assistant",
        content=[
            ClaudeContentBlockToolUse(
                type="tool_use",
                id="call-1",
                name="read_file",
                input={"path": "/tmp/test.txt"},
            )
        ],
        model="claude-3-5-haiku-20241022",
        stop_reason="tool_use",
        usage=ClaudeUsage(input_tokens=12, output_tokens=7),
    )

    with patch(
        "src.api.endpoints.BoostOrchestrator.execute_with_boost",
        new=AsyncMock(return_value=boost_response),
    ) as mock_execute:
        response = await test_client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 200,
                "messages": [
                    {
                        "role": "user",
                        "content": "Read the file /tmp/test.txt and tell me what's in it",
                    }
                ],
                "tools": [
                    {
                        "name": "read_file",
                        "description": "Read a file from filesystem",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Path of the file",
                                }
                            },
                            "required": ["path"],
                        },
                    }
                ],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["content"][0]["type"] == "tool_use"
    assert body["content"][0]["name"] == "read_file"
    assert mock_execute.await_count == 1


@pytest.mark.asyncio
async def test_boost_mode_with_sonnet(test_client, monkeypatch):
    """Verify sonnet tier models also use the boost orchestrator when enabled."""
    _boost_available(monkeypatch, "MIDDLE_MODEL")

    boost_response = ClaudeMessageResponse(
        id="msg_sonnet",
        type="message",
        role="assistant",
        content=[ClaudeContentBlockText(type="text", text="Paris is the capital of France.")],
        model="claude-3-5-sonnet-20241022",
        stop_reason="end_turn",
        usage=ClaudeUsage(input_tokens=14, output_tokens=6),
    )

    with patch(
        "src.api.endpoints.BoostOrchestrator.execute_with_boost",
        new=AsyncMock(return_value=boost_response),
    ) as mock_execute:
        response = await test_client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
                "tools": [
                    {
                        "name": "search_web",
                        "description": "Search web for information",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query"}
                            },
                            "required": ["query"],
                        },
                    }
                ],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["content"][0]["text"] == "Paris is the capital of France."
    await_args = mock_execute.await_args_list[0].args
    assert await_args[0].model == "claude-3-5-sonnet-20241022"


@pytest.mark.asyncio
async def test_boost_mode_fallback_to_direct_execution(test_client, monkeypatch):
    """When boost is disabled, the route should fall back to direct OpenAI execution."""
    monkeypatch.setattr(config, "enable_boost_support", "NONE")

    openai_payload = {
        "choices": [
            {
                "message": {"content": "Direct path response", "tool_calls": []},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }

    with (
        patch(
            "src.api.endpoints.openai_client.create_chat_completion",
            new=AsyncMock(return_value=openai_payload),
        ) as mock_completion,
        patch(
            "src.api.endpoints.convert_openai_to_claude_response",
            return_value={"content": [{"type": "text", "text": "Direct path response"}]},
        ) as mock_convert,
    ):
        response = await test_client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello, world"}],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["content"][0]["text"] == "Direct path response"
    mock_completion.assert_awaited_once()
    mock_convert.assert_called_once()
