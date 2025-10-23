"""Test script for boost mode functionality only."""

import asyncio
import json
import httpx
from dotenv import load_dotenv

load_dotenv()


async def test_boost_mode_summary_response():
    """Test boost mode with SUMMARY response (no tools needed)."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8083/v1/messages",
            json={
                "model": "claude-3-5-haiku-20241022",  # Should trigger boost if enabled for SMALL_MODEL
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "What is 2 + 2?"}
                ],
                "tools": [
                    {
                        "name": "calculator",
                        "description": "Perform basic arithmetic calculations",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "Mathematical expression to calculate"
                                }
                            },
                            "required": ["expression"]
                        }
                    }
                ]
            }
        )

        print("\nBoost mode SUMMARY response:")
        result = response.json()
        print(json.dumps(result, indent=2))

        # Verify response structure
        assert "content" in result
        assert len(result["content"]) > 0
        assert result["content"][0]["type"] == "text"
        print("✅ SUMMARY response test passed")


async def test_boost_mode_guidance_response():
    """Test boost mode with GUIDANCE response (tools needed)."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8083/v1/messages",
            json={
                "model": "claude-3-5-haiku-20241022",  # Should trigger boost if enabled for SMALL_MODEL
                "max_tokens": 200,
                "messages": [
                    {"role": "user", "content": "Read the file /tmp/test.txt and tell me what's in it"}
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
                                    "description": "Path to file to read"
                                }
                            },
                            "required": ["path"]
                        }
                    }
                ]
            }
        )

        print("\nBoost mode GUIDANCE response:")
        result = response.json()
        print(json.dumps(result, indent=2))

        # Verify response structure
        assert "content" in result
        print("✅ GUIDANCE response test passed")


async def test_boost_mode_with_sonnet():
    """Test boost mode with sonnet model (should use boost if enabled for MIDDLE_MODEL)."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8083/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",  # Should trigger boost if enabled for MIDDLE_MODEL
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                "tools": [
                    {
                        "name": "search_web",
                        "description": "Search web for information",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                ]
            }
        )

        print("\nBoost mode with sonnet (MIDDLE_MODEL):")
        result = response.json()
        print(json.dumps(result, indent=2))

        # Verify response structure
        assert "content" in result
        print("✅ Sonnet boost mode test passed")


async def test_boost_mode_fallback():
    """Test boost mode fallback to direct execution."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8083/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",  # Should use direct path if boost not enabled for MIDDLE_MODEL
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "Hello, this should use direct execution"}
                ],
                "tools": [
                    {
                        "name": "test_tool",
                        "description": "Test tool",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "param": {
                                    "type": "string",
                                    "description": "Test parameter"
                                }
                            },
                            "required": ["param"]
                        }
                    }
                ]
            }
        )

        print("\nBoost mode fallback test:")
        result = response.json()
        print(json.dumps(result, indent=2))

        # Verify we get a valid response
        assert "content" in result
        assert response.status_code == 200
        print("✅ Fallback test passed")


async def main():
    """Run boost mode tests."""
    print("🚀 Testing Boost Mode Features")
    print("=" * 50)

    try:
        await test_boost_mode_summary_response()
        await test_boost_mode_guidance_response()
        await test_boost_mode_with_sonnet()
        await test_boost_mode_fallback()

        print("\n✅ All boost mode tests passed!")

    except Exception as e:
        print(f"\n❌ Boost test failed: {e}")
        import traceback
        traceback.print_exc()
        print("Make sure server is running with boost configuration enabled")


if __name__ == "__main__":
    asyncio.run(main())