from typing import Dict, Any, List, Optional
from src.core.logging import logger

class AuxiliaryModelBuilder:
    """Builds requests for the auxiliary model based on boost guidance"""

    @staticmethod
    def build_auxiliary_request(
        original_request: Any,
        analysis: str,
        guidance: str,
        tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build a request for the auxiliary model that includes:
        - Original user request
        - Boost model's analysis
        - Boost model's guidance
        - Actual tools parameter for execution
        """
        # Convert original request to dict if needed
        if hasattr(original_request, 'model'):
            model = original_request.model
            messages = original_request.messages
            stream = getattr(original_request, 'stream', False)
            max_tokens = getattr(original_request, 'max_tokens', None)
            temperature = getattr(original_request, 'temperature', None)
        else:
            # Already a dict
            model = original_request.get('model')
            messages = original_request.get('messages', [])
            stream = original_request.get('stream', False)
            max_tokens = original_request.get('max_tokens')
            temperature = original_request.get('temperature')

        # Create enhanced messages that include boost guidance
        enhanced_messages = []

        # Add system message with boost guidance
        system_content = f"""You are an AI assistant helping with a user request. The boost model has provided the following analysis and guidance:

ANALYSIS:
{analysis}

GUIDANCE:
{guidance}

Please follow the guidance to complete the user's request. Use the available tools as instructed."""

        enhanced_messages.append({
            "role": "system",
            "content": system_content
        })

        # Add original messages (excluding any existing system message)
        for msg in messages:
            if msg.get('role') != 'system':
                enhanced_messages.append(msg)

        # Build the auxiliary request
        auxiliary_request = {
            "model": model,
            "messages": enhanced_messages,
            "stream": stream,
            "tools": tools,  # Actual tools parameter for execution
            "tool_choice": "auto"  # Let the model decide when to use tools
        }

        # Add optional parameters if present
        if max_tokens is not None:
            auxiliary_request["max_tokens"] = max_tokens
        if temperature is not None:
            auxiliary_request["temperature"] = temperature

        logger.info(f"Built auxiliary request with {len(tools)} tools")
        return auxiliary_request

    @staticmethod
    def detect_tool_usage(response: Dict[str, Any]) -> bool:
        """
        Detect if the auxiliary model actually used tools in its response
        """
        # Check for tool_calls in the response
        if 'choices' in response and response['choices']:
            message = response['choices'][0].get('message', {})
            if 'tool_calls' in message and message['tool_calls']:
                return True

        # Check streaming response for tool calls
        if 'delta' in response:
            delta = response['delta']
            if 'tool_calls' in delta and delta['tool_calls']:
                return True

        return False

    @staticmethod
    def extract_final_response(response: Dict[str, Any]) -> str:
        """
        Extract the final text response from the auxiliary model
        """
        if 'choices' in response and response['choices']:
            message = response['choices'][0].get('message', {})
            content = message.get('content', '')
            if content:
                return content

            # If no content but there were tool calls, we might need to handle differently
            if 'tool_calls' in message:
                # This is a tool-only response, may need further processing
                logger.warning("Auxiliary model returned only tool calls without content")
                return ""

        return ""