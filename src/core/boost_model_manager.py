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

        # Use custom template if provided and valid, otherwise use default
        custom_template = getattr(self.config, "boost_wrapper_template", None)
        if isinstance(custom_template, str) and custom_template.strip():
            template = custom_template
        else:
            template = self._get_default_wrapper_template()

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

        summary = self._extract_section(response, "SUMMARY:")
        analysis = self._extract_section(response, "ANALYSIS:")
        guidance = self._extract_section(response, "GUIDANCE:")

        if summary:
            return "SUMMARY", analysis or "", summary
        if guidance:
            return "GUIDANCE", analysis or "", guidance
        if analysis:
            return "OTHER", analysis, ""
        return "OTHER", "", ""

    def _extract_section(self, text: str, section_name: str) -> Optional[str]:
        """Extract a section from the boost model response"""
        lines = text.split('\n')
        in_section = False
        section_lines = []

        for line in lines:
            stripped = line.strip()

            if stripped.startswith(section_name):
                in_section = True
                # Get content after the section header
                content = stripped[len(section_name):].strip()
                if content:
                    section_lines.append(content)
            elif in_section:
                if stripped.startswith("---") or stripped.startswith(("SUMMARY:", "ANALYSIS:", "GUIDANCE:")):
                    break
                section_lines.append(line.rstrip())

        # Trim leading/trailing empty lines while preserving intentional spacing within the section
        while section_lines and section_lines[0] == "":
            section_lines.pop(0)
        while section_lines and section_lines[-1] == "":
            section_lines.pop()

        return '\n'.join(section_lines) if section_lines else None

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
