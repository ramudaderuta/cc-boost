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
        # Validate request
        if not claude_request or not hasattr(claude_request, 'model') or not claude_request.model:
            return self._create_error_response("Invalid request: missing model", claude_request)

        logger.info(f"Starting boost execution for model: {claude_request.model}")

        # Convert to OpenAI format to extract tools
        try:
            openai_request = convert_claude_to_openai(claude_request, model_manager)
            tools = openai_request.get('tools', [])
        except Exception as e:
            logger.error(f"Failed to convert request: {e}")
            return self._create_error_response(f"Request conversion failed: {str(e)}", claude_request)

        # Initialize loop state
        loop_state = LoopState()

        # Extract user request from messages
        user_request = self._extract_user_request(claude_request.messages)

        while loop_state.can_continue():
            logger.info(f"Boost loop iteration: {loop_state.loop_count}")

            # Get guidance from boost model
            try:
                response_type, analysis, guidance = await self.boost_manager.get_boost_guidance(
                    user_request=user_request,
                    tools=tools,
                    loop_count=loop_state.loop_count,
                    previous_attempts=loop_state.previous_attempts,
                )
            except Exception as exc:
                logger.error(f"Boost model call failed: {exc}")
                loop_state.add_attempt(f"Boost model error: {str(exc)}")
                if loop_state.increment_loop():
                    continue
                logger.warning(f"Max loops ({loop_state.max_loops}) reached")
                return self._create_error_response("Maximum retry attempts reached", claude_request)

            logger.info(f"Boost model response type: {response_type}")

            if response_type == "SUMMARY":
                logger.info("Boost model provided SUMMARY response")
                loop_state.register_analysis(analysis)
                return self._create_final_claude_response(guidance, claude_request)

            if response_type == "GUIDANCE":
                loop_state.register_analysis(analysis)
                guidance_is_new = loop_state.register_guidance(guidance)

                if not guidance_is_new and loop_state.loop_count > 0:
                    logger.warning("Boost guidance repeated without progress; exiting loop early.")
                    return self._create_error_response("Repeated guidance detected without progress", claude_request)

                auxiliary_request = AuxiliaryModelBuilder.build_auxiliary_request(
                    claude_request,
                    analysis,
                    guidance,
                    tools,
                )

                try:
                    if claude_request.stream:
                        return await self._handle_streaming_auxiliary(
                            auxiliary_request, claude_request, request_id, loop_state
                        )

                    tools_used, claude_response, auxiliary_text = await self._handle_non_streaming_auxiliary(
                        auxiliary_request,
                        claude_request,
                        request_id,
                        loop_state,
                    )

                    if tools_used:
                        return claude_response

                    loop_state.add_attempt(
                        f"Auxiliary model didn't use tools. Response: {auxiliary_text[:200]}..."
                    )
                    if loop_state.increment_loop():
                        continue
                    logger.warning(f"Max loops ({loop_state.max_loops}) reached")
                    return self._create_error_response(
                        "Auxiliary model failed to use tools after multiple attempts",
                        claude_request,
                    )

                except Exception as exc:
                    logger.error(f"Auxiliary model execution failed: {exc}")
                    loop_state.add_attempt(f"Auxiliary execution failed: {str(exc)}")
                    if loop_state.increment_loop():
                        continue
                    logger.warning(f"Max loops ({loop_state.max_loops}) reached")
                    return self._create_error_response(
                        "Auxiliary execution failed after multiple attempts",
                        claude_request,
                    )

            # OTHER response or unhandled case
            logger.warning("Boost model returned invalid format, retrying...")
            loop_state.register_analysis(analysis)
            loop_state.add_attempt(f"Invalid response format: {analysis[:200]}...")
            if loop_state.increment_loop():
                continue
            logger.warning(f"Max loops ({loop_state.max_loops}) reached")
            return self._create_error_response("Maximum retry attempts reached", claude_request)

        logger.warning(f"Max loops ({loop_state.max_loops}) reached")
        return self._create_error_response("Maximum retry attempts reached", claude_request)

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
                                text_parts.append(block.text.strip())
                        return ' '.join(part for part in text_parts if part).strip()
            elif isinstance(message, dict) and message.get('role') == 'user':
                content = message.get('content', '')
                if isinstance(content, str):
                    return content.strip()
                elif isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get('type') == 'text':
                            text_parts.append((block.get('text') or '').strip())
                    return ' '.join(part for part in text_parts if part).strip()

        return "User request not found"

    async def _handle_non_streaming_auxiliary(
        self,
        auxiliary_request: Dict[str, Any],
        original_claude_request: Any,
        request_id: str,
        loop_state: LoopState,
    ) -> Tuple[bool, Optional[Any], str]:
        """Handle non-streaming auxiliary model execution."""
        openai_response = await self.openai_client.create_chat_completion(
            auxiliary_request, request_id
        )

        # Check if tools were used
        if AuxiliaryModelBuilder.detect_tool_usage(openai_response):
            logger.info("Auxiliary model used tools successfully")
            claude_response = convert_openai_to_claude_response(openai_response, original_claude_request)
            return True, claude_response, ""

        logger.warning("Auxiliary model did not use tools, triggering loop")
        content = AuxiliaryModelBuilder.extract_final_response(openai_response)
        return False, None, content or ""

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
        openai_stream = await self.openai_client.create_chat_completion_stream(
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

        # Get model name from request or use default
        model_name = getattr(original_request, 'model', None) or "unknown"

        response = ClaudeMessageResponse(
            id="error-" + str(hash(error_message) % 1000000),
            type="message",
            role="assistant",
            content=[ClaudeContentBlockText(type="text", text=f"Error: {error_message}")],
            model=model_name,
            stop_reason="end_turn",
            stop_sequence=None,
            usage=ClaudeUsage(
                input_tokens=0,
                output_tokens=0
            )
        )

        return response
