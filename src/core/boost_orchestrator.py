import asyncio
from typing import Any, Dict, List, Optional, Tuple
from src.core.boost_model_manager import BoostModelManager
from src.core.loop_controller import LoopState
from src.core.auxiliary_builder import AuxiliaryModelBuilder
from src.core.client import OpenAIClient
from src.core.logging import logger
from src.conversion.response_converter import convert_openai_to_claude_response
from src.conversion.request_converter import convert_claude_to_openai
from src.core.model_manager import model_manager
from src.models.claude import ClaudeMessageResponse, ClaudeUsage, ClaudeContentBlockText

class BoostOrchestrator:
    """Orchestrates the boost-directed tool-calling flow"""

    def __init__(self, config, openai_client: OpenAIClient):
        self.config = config
        self.openai_client = openai_client
        self.boost_manager = BoostModelManager(config)

    async def execute_with_boost(
        self,
        claude_request: Any,
        request_id: str
    ) -> Any:
        """
        Execute a request using the boost-directed tool-calling flow

        Returns:
            Claude-format response
        """
        logger.info(f"Starting boost execution for model: {claude_request.model}")

        # Convert to OpenAI format to extract tools
        openai_request = convert_claude_to_openai(claude_request, model_manager)
        tools = openai_request.get('tools', [])

        # Initialize loop state
        loop_state = LoopState()

        # Extract user request from messages
        user_request = self._extract_user_request(claude_request.messages)

        try:
            while loop_state.can_continue():
                logger.info(f"Boost loop iteration: {loop_state.loop_count}")

                # Get guidance from boost model
                response_type, analysis, guidance = await self.boost_manager.get_boost_guidance(
                    user_request=user_request,
                    tools=tools,
                    loop_count=loop_state.loop_count,
                    previous_attempts=loop_state.previous_attempts
                )

                logger.info(f"Boost model response type: {response_type}")

                if response_type == "SUMMARY":
                    # Boost model provided final answer, no auxiliary needed
                    logger.info("Boost model provided SUMMARY response")
                    return self._create_final_claude_response(guidance, claude_request)

                elif response_type == "GUIDANCE":
                    # Build auxiliary request with guidance
                    auxiliary_request = AuxiliaryModelBuilder.build_auxiliary_request(
                        claude_request,
                        analysis,
                        guidance,
                        tools
                    )

                    # Execute with auxiliary model
                    try:
                        if claude_request.stream:
                            # Handle streaming response
                            return await self._handle_streaming_auxiliary(
                                auxiliary_request, claude_request, request_id, loop_state
                            )
                        else:
                            # Handle non-streaming response
                            return await self._handle_non_streaming_auxiliary(
                                auxiliary_request, claude_request, request_id, loop_state
                            )

                    except Exception as e:
                        logger.error(f"Auxiliary model execution failed: {e}")
                        loop_state.add_attempt(f"Auxiliary execution failed: {str(e)}")
                        if not loop_state.increment_loop():
                            return self._create_error_response("Execution failed after multiple attempts", claude_request)

                else:  # OTHER
                    # Invalid response format, retry
                    logger.warning(f"Boost model returned invalid format, retrying...")
                    loop_state.add_attempt(f"Invalid response format: {analysis[:200]}...")
                    if not loop_state.increment_loop():
                        return self._create_error_response("Boost model failed to provide valid guidance", claude_request)

            # Max loops reached
            logger.warning(f"Max loops ({loop_state.max_loops}) reached")
            return self._create_error_response("Maximum retry attempts reached", claude_request)

        finally:
            await self.boost_manager.close()

    def _extract_user_request(self, messages: List[Any]) -> str:
        """Extract the user's request from message history"""
        for message in reversed(messages):
            if hasattr(message, 'role') and message.role == 'user':
                if hasattr(message, 'content'):
                    if isinstance(message.content, str):
                        return message.content
                    elif isinstance(message.content, list):
                        # Extract text from content blocks
                        text_parts = []
                        for block in message.content:
                            if hasattr(block, 'text') and block.text:
                                text_parts.append(block.text)
                        return ' '.join(text_parts)
            elif isinstance(message, dict) and message.get('role') == 'user':
                content = message.get('content', '')
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get('type') == 'text':
                            text_parts.append(block.get('text', ''))
                    return ' '.join(text_parts)

        return "User request not found"

    async def _handle_non_streaming_auxiliary(
        self,
        auxiliary_request: Dict[str, Any],
        original_claude_request: Any,
        request_id: str,
        loop_state: LoopState
    ) -> Any:
        """Handle non-streaming auxiliary model execution"""
        openai_response = await self.openai_client.create_chat_completion(
            auxiliary_request, request_id
        )

        # Check if tools were used
        if AuxiliaryModelBuilder.detect_tool_usage(openai_response):
            logger.info("Auxiliary model used tools successfully")
            # Convert to Claude format and return
            return convert_openai_to_claude_response(openai_response, original_claude_request)
        else:
            # No tools used, increment loop and retry
            logger.warning("Auxiliary model did not use tools, triggering loop")
            content = AuxiliaryModelBuilder.extract_final_response(openai_response)
            loop_state.add_attempt(f"Auxiliary model didn't use tools. Response: {content[:200]}...")
            if loop_state.increment_loop():
                # Recursively retry with boost
                return await self.execute_with_boost(original_claude_request, request_id)
            else:
                return self._create_error_response("Auxiliary model failed to use tools after multiple attempts", original_claude_request)

    async def _handle_streaming_auxiliary(
        self,
        auxiliary_request: Dict[str, Any],
        original_claude_request: Any,
        request_id: str,
        loop_state: LoopState
    ):
        """Handle streaming auxiliary model execution"""
        from src.conversion.response_converter import convert_openai_streaming_to_claude_with_cancellation
        from fastapi.responses import StreamingResponse

        # Create the streaming response
        openai_stream = self.openai_client.create_chat_completion_stream(
            auxiliary_request, request_id
        )

        # We need to monitor the stream for tool usage
        # This is complex for streaming - for now, we'll wrap the stream
        async def monitored_stream():
            tool_usage_detected = False
            stream_content = []

            async for chunk in convert_openai_streaming_to_claude_with_cancellation(
                openai_stream,
                original_claude_request,
                logger,
                None,  # No HTTP request object available here
                self.openai_client,
                request_id,
            ):
                # Check if this chunk indicates tool usage
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'tool_calls'):
                    tool_usage_detected = True

                # Also collect content for potential retry
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'content'):
                    stream_content.append(chunk.delta.content)

                yield chunk

            # After stream ends, check if tools were used
            if not tool_usage_detected:
                logger.warning("Streaming response did not use tools")
                # This is tricky with streaming - we've already sent the response
                # In a full implementation, we might need client-side retry logic
                # For now, we'll log the issue

        return StreamingResponse(
            monitored_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )

    def _create_final_claude_response(self, final_text: str, original_request: Any) -> Any:
        """Create a Claude-format response from boost model's SUMMARY response"""
        from src.models.claude import ClaudeMessageResponse, ClaudeUsage, ClaudeContentBlockText

        # Create a minimal Claude response
        response = ClaudeMessageResponse(
            id="boost-" + str(hash(final_text) % 1000000),
            type="message",
            role="assistant",
            content=[ClaudeContentBlockText(type="text", text=final_text)],
            model=original_request.model,
            stop_reason="end_turn",
            stop_sequence=None,
            usage=ClaudeUsage(
                input_tokens=0,  # We don't have exact counts
                output_tokens=0
            )
        )

        return response

    def _create_error_response(self, error_message: str, original_request: Any) -> Any:
        """Create an error response in Claude format"""
        from src.models.claude import ClaudeMessageResponse, ClaudeUsage, ClaudeContentBlockText

        response = ClaudeMessageResponse(
            id="error-" + str(hash(error_message) % 1000000),
            type="message",
            role="assistant",
            content=[ClaudeContentBlockText(type="text", text=f"Error: {error_message}")],
            model=original_request.model,
            stop_reason="end_turn",
            stop_sequence=None,
            usage=ClaudeUsage(
                input_tokens=0,
                output_tokens=0
            )
        )

        return response