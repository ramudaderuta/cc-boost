import httpx
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from src.core.config import config

logger = logging.getLogger(__name__)

class BoostModelManager:
    """Manages communication with the boost model for tool-directed execution"""

    def __init__(self, config):
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.boost_base_url,
            timeout=config.request_timeout,
            headers={
                "Authorization": f"Bearer {config.boost_api_key}",
                "Content-Type": "application/json"
            }
        )

    def _get_default_wrapper_template(self) -> str:
        """Get the default wrapper template for boost model"""
        return """You are a boost model assisting an auxiliary model. Your response MUST follow ONE of these three formats:

FORMAT 1 - FINAL RESPONSE (when no tools needed):
SUMMARY:
[Provide the final answer directly without using auxiliary models]

FORMAT 2 - GUIDANCE FOR AUXILIARY MODEL (when tools needed):
ANALYSIS:
[Reasoning and understanding of the request context (trace the context, uncertainties, and potential solution paths sequentially, refining thoughts while keeping continuity)]

GUIDANCE:
[Instructions for the auxiliary model's tasks (include which tools to call and what operations to perform, and the content of the operations should be detailed)]

FORMAT 3 - OTHER (any other response will trigger a loop retry):
[Any response that doesn't match FORMAT 1 or 2]

---
Current ReAct Loop: {loop_count}
Previous Attempts: {previous_attempts}

User Request: {user_request}

Available Tools:
{tools_text}"""

    def _format_tools_for_message(self, tools: List[Dict[str, Any]]) -> str:
        """Convert tool definitions to text format for boost model"""
        if not tools:
            return "No tools available"

        tools_text = []
        for tool in tools:
            name = tool.get('name', 'unknown')
            description = tool.get('description', 'No description')

            # Extract parameters
            params = tool.get('input_schema', {})
            if isinstance(params, dict):
                properties = params.get('properties', {})
                required = params.get('required', [])

                param_details = []
                for param_name, param_info in properties.items():
                    param_type = param_info.get('type', 'string')
                    param_desc = param_info.get('description', '')
                    req_marker = " (required)" if param_name in required else " (optional)"
                    param_details.append(f"- {param_name}: {param_type}{req_marker} - {param_desc}")

                tools_text.append(f"- {name}: {description}. Parameters: {', '.join(param_details)}")
            else:
                tools_text.append(f"- {name}: {description}")

        return '\n'.join(tools_text)

    def build_boost_message(
        self,
        user_request: str,
        tools: List[Dict[str, Any]],
        loop_count: int = 0,
        previous_attempts: List[str] = None
    ) -> Dict[str, Any]:
        """Build message for boost model with tools embedded in content"""

        # Use custom template if provided, otherwise use default
        template = self.config.boost_wrapper_template or self._get_default_wrapper_template()

        # Format tools as text
        tools_text = self._format_tools_for_message(tools)

        # Format previous attempts
        previous_attempts_text = ""
        if previous_attempts:
            previous_attempts_text = "\n".join([f"- {attempt}" for attempt in previous_attempts])

        # Build the complete message
        message_content = template.format(
            loop_count=loop_count,
            previous_attempts=previous_attempts_text or "None",
            user_request=user_request,
            tools_text=tools_text
        )

        return {
            "model": self.config.boost_model,
            "messages": [
                {
                    "role": "user",
                    "content": message_content
                }
            ],
            "temperature": 0.7,
            "max_tokens": 4096
            # Note: NO tools parameter - tools are embedded in message content
        }

    async def call_boost_model(self, message: Dict[str, Any]) -> str:
        """Call the boost model and return the response text"""
        try:
            logger.info(f"Calling boost model: {self.config.boost_model}")

            response = await self.client.post(
                "/chat/completions",
                json=message
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            logger.info(f"Boost model response received: {len(content)} characters")
            return content

        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling boost model: {e}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"Invalid response format from boost model: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling boost model: {e}")
            raise

    async def get_boost_guidance(
        self,
        user_request: str,
        tools: List[Dict[str, Any]],
        loop_count: int = 0,
        previous_attempts: List[str] = None
    ) -> Tuple[str, str, str]:
        """
        Get guidance from boost model.

        Returns:
            Tuple of (response_type, analysis, guidance)
            - response_type: 'SUMMARY', 'GUIDANCE', or 'OTHER'
            - analysis: Content from ANALYSIS section (empty if not present)
            - guidance: Content from GUIDANCE section (empty if not present)
        """
        message = self.build_boost_message(user_request, tools, loop_count, previous_attempts)
        response = await self.call_boost_model(message)

        # Parse the response
        response_type = "OTHER"
        analysis = ""
        guidance = ""

        lines = response.split('\n')
        current_section = None
        section_content = []

        for line in lines:
            line = line.strip()

            # Check for section headers
            if line.startswith("SUMMARY:"):
                response_type = "SUMMARY"
                current_section = "final"
                section_content = [line[8:].strip()]  # Content after "SUMMARY:"
                continue
            elif line.startswith("ANALYSIS:"):
                current_section = "analysis"
                section_content = [line[9:].strip()]  # Content after "ANALYSIS:"
                continue
            elif line.startswith("GUIDANCE:"):
                current_section = "guidance"
                section_content = [line[9:].strip()]  # Content after "GUIDANCE:"
                continue
            elif line.startswith("---"):
                # End of structured content
                break

            # Add content to current section
            if current_section and line:
                section_content.append(line)

        # Extract section contents
        if response_type == "SUMMARY":
            guidance = '\n'.join(section_content)
        else:
            # For non-SUMMARY responses, check if we have ANALYSIS/GUIDANCE
            # Re-parse to extract these sections properly
            analysis_match = self._extract_section(response, "ANALYSIS:")
            guidance_match = self._extract_section(response, "GUIDANCE:")

            if guidance_match:
                response_type = "GUIDANCE"
                guidance = guidance_match
                analysis = analysis_match or ""
            elif analysis_match and not guidance_match:
                # Has analysis but no guidance - treat as OTHER
                response_type = "OTHER"
                analysis = analysis_match

        return response_type, analysis, guidance

    def _extract_section(self, text: str, section_name: str) -> Optional[str]:
        """Extract a section from the boost model response"""
        lines = text.split('\n')
        in_section = False
        section_lines = []

        for line in lines:
            if line.strip().startswith(section_name):
                in_section = True
                # Get content after the section header
                content = line.strip()[len(section_name):].strip()
                if content:
                    section_lines.append(content)
            elif in_section:
                if line.strip().startswith("---") or line.strip().startswith(("SUMMARY:", "ANALYSIS:", "GUIDANCE:")):
                    break
                section_lines.append(line)

        return '\n'.join(section_lines) if section_lines else None

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()