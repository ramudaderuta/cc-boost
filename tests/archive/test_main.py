"""Functional tests that exercise the public HTTP surface of the proxy."""

from unittest.mock import AsyncMock, patch

import pytest

from src.core.config import config


@pytest.mark.asyncio
async def test_basic_chat_path_invokes_openai_client(test_client, monkeypatch):
    """A plain chat request should call the OpenAI client when boost is disabled."""
    monkeypatch.setattr(config, "enable_boost_support", "NONE")

    openai_payload = {
        "choices": [
            {
                "message": {"content": "Hello! I'm doing well.", "tool_calls": []},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 6},
    }

    calls = []

    async def fake_completion(request, request_id):
        calls.append((request, request_id))
        return openai_payload

    with (
        patch(
            "src.api.endpoints.openai_client.create_chat_completion",
            new=fake_completion,
        ),
        patch(
            "src.api.endpoints.convert_openai_to_claude_response",
            return_value={"content": [{"type": "text", "text": "Hello! I'm doing well."}]},
        ) as mock_convert,
    ):
        response = await test_client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["content"][0]["text"] == "Hello! I'm doing well."
    assert len(calls) == 1
    mock_convert.assert_called_once()


@pytest.mark.asyncio
async def test_streaming_chat_emits_custom_events(test_client, monkeypatch):
    """Streaming requests should surface events produced by the converter."""
    monkeypatch.setattr(config, "enable_boost_support", "NONE")

    async def fake_openai_stream():
        yield "data: {\"choices\": []}"
        yield "data: [DONE]"

    async def fake_create_stream(*_args, **_kwargs):
        async for item in fake_openai_stream():
            yield item

    async def fake_converter(*_args, **_kwargs):
        yield "event: message.start\ndata: {}\n\n"
        yield "event: message.delta\ndata: {\"text\": \"Hi\"}\n\n"
        yield "event: message.stop\ndata: {}\n\n"

    with (
        patch(
            "src.api.endpoints.openai_client.create_chat_completion_stream",
            new=fake_create_stream,
        ),
        patch(
            "src.api.endpoints.convert_openai_streaming_to_claude_with_cancellation",
            side_effect=fake_converter,
        ),
    ):
        async with test_client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 150,
                "messages": [{"role": "user", "content": "Tell me a short joke"}],
                "stream": True,
            },
        ) as response:
            events = [line async for line in response.aiter_lines() if line.strip()]

    assert response.status_code == 200
    assert events == [
        "event: message.start",
        "data: {}",
        "event: message.delta",
        "data: {\"text\": \"Hi\"}",
        "event: message.stop",
        "data: {}",
    ]


@pytest.mark.asyncio
async def test_function_calling_returns_tool_use_block(test_client, monkeypatch):
    """Tool-calling responses should be converted to Claude tool_use blocks."""
    monkeypatch.setattr(config, "enable_boost_support", "NONE")

    openai_payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "New York", "unit": "celsius"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 42, "completion_tokens": 0},
    }

    async def fake_completion(request, request_id):
        return openai_payload

    with patch(
        "src.api.endpoints.openai_client.create_chat_completion",
        new=fake_completion,
    ):
        response = await test_client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 200,
                "messages": [
                    {
                        "role": "user",
                        "content": "What's the weather like in New York? Please use the weather function.",
                    }
                ],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get the current weather for a location",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                            },
                            "required": ["location"],
                        },
                    }
                ],
                "tool_choice": {"type": "auto"},
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["content"][0]["type"] == "tool_use"
    assert body["content"][0]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_system_message_flow(test_client, monkeypatch):
    """System messages should propagate to the OpenAI client."""
    monkeypatch.setattr(config, "enable_boost_support", "NONE")

    openai_payload = {
        "choices": [
            {
                "message": {"content": "AI is the simulation of intelligence.", "tool_calls": []},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 6},
    }

    calls = []

    async def fake_completion(request, request_id):
        calls.append((request, request_id))
        return openai_payload

    with (
        patch(
            "src.api.endpoints.openai_client.create_chat_completion",
            new=fake_completion,
        ),
        patch(
            "src.api.endpoints.convert_openai_to_claude_response",
            return_value={
                "content": [{"type": "text", "text": "AI is the simulation of intelligence."}]
            },
        ),
    ):
        response = await test_client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 100,
                "system": "You are a helpful assistant that always responds in haiku format.",
                "messages": [{"role": "user", "content": "Explain what AI is"}],
            },
        )

    assert response.status_code == 200
    request_payload = calls[0][0]
    assert request_payload["messages"][0]["role"] == "system"
    assert "haiku format" in request_payload["messages"][0]["content"]


@pytest.mark.asyncio
async def test_multimodal_input_pass_through(test_client, monkeypatch):
    """Ensure multimodal requests are accepted and forwarded."""
    monkeypatch.setattr(config, "enable_boost_support", "NONE")

    openai_payload = {
        "choices": [
            {
                "message": {"content": "It appears to be a transparent pixel.", "tool_calls": []},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8},
    }

    async def fake_completion(request, request_id):
        return openai_payload

    with patch(
        "src.api.endpoints.openai_client.create_chat_completion",
        new=fake_completion,
    ), patch(
        "src.api.endpoints.convert_openai_to_claude_response",
        return_value={"content": [{"type": "text", "text": "It appears to be a transparent pixel."}]},
    ):
        response = await test_client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What do you see in this image?"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU8PJAAAAASUVORK5CYII=",
                                },
                            },
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert response.json()["content"][0]["text"] == "It appears to be a transparent pixel."


@pytest.mark.asyncio
async def test_conversation_with_tool_use_flow(test_client, monkeypatch):
    """Simulate a two-step tool interaction via the HTTP interface."""
    monkeypatch.setattr(config, "enable_boost_support", "NONE")

    first_openai_payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_calc",
                            "type": "function",
                            "function": {"name": "calculator", "arguments": '{"expression": "25 * 4"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 0},
        "id": "chatcmpl-first",
    }
    second_openai_payload = {
        "choices": [
            {
                "message": {"content": "25 * 4 equals 100.", "tool_calls": []},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 18, "completion_tokens": 6},
        "id": "chatcmpl-second",
    }

    calls = []
    payload_iter = iter([first_openai_payload, second_openai_payload])

    async def fake_completion(request, request_id):
        calls.append((request, request_id))
        return next(payload_iter)

    with patch(
        "src.api.endpoints.openai_client.create_chat_completion",
        new=fake_completion,
    ):
        response1 = await test_client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": "Calculate 25 * 4 using the calculator tool"}],
                "tools": [
                    {
                        "name": "calculator",
                        "description": "Perform basic arithmetic calculations",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "Mathematical expression to calculate",
                                }
                            },
                            "required": ["expression"],
                        },
                    }
                ],
            },
        )

        first_body = response1.json()
        tool_use = next(block for block in first_body["content"] if block["type"] == "tool_use")

        response2 = await test_client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "Calculate 25 * 4 using the calculator tool"},
                    {"role": "assistant", "content": first_body["content"]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": tool_use["id"], "content": "100"}
                        ],
                    },
                ],
            },
        )

    assert response1.status_code == 200
    assert tool_use["name"] == "calculator"
    assert response2.status_code == 200
    assert response2.json()["content"][0]["text"] == "25 * 4 equals 100."
    assert len(calls) == 2
