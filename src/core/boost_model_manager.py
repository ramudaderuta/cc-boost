import asyncio
import hashlib
import httpx
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock as ThreadLock
from time import monotonic
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached boost response with TTL support."""

    response_type: str
    analysis: str
    payload: str
    raw_response: str
    created_at: float


class BoostModelManager:
    """Manages communication with the boost model for tool-directed execution."""

    _section_cache_limit = 6
    _response_cache_limit = 32
    _response_cache_ttl = 60.0  # seconds
    _client_pool: Dict[str, httpx.AsyncClient] = {}
    _pool_lock: ThreadLock = ThreadLock()

    def __init__(self, config):
        self.config = config
        key_material = f"{self.config.boost_base_url}|{self.config.boost_model}|{self.config.request_timeout}|{self.config.boost_api_key}"
        self._pool_key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
        self.client = self._ensure_client()
        self._section_cache: "OrderedDict[str, Dict[str, Optional[str]]]" = OrderedDict()
        self._response_cache: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._response_cache_lock: Optional[asyncio.Lock] = None

    def _get_default_wrapper_template(self) -> str:
        """Get the default wrapper template for boost model."""
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
        """Convert tool definitions to text format for boost model."""
        if not tools:
            return "No tools available"

        tools_text = []
        for tool in tools:
            name = tool.get("name", "unknown")
            description = tool.get("description", "No description")

            params = tool.get("input_schema", {})
            if isinstance(params, dict):
                properties = params.get("properties", {})
                required = params.get("required", [])

                param_details = []
                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "string")
                    param_desc = param_info.get("description", "")
                    req_marker = " (required)" if param_name in required else " (optional)"
                    param_details.append(f"- {param_name}: {param_type}{req_marker} - {param_desc}")

                tools_text.append(f"- {name}: {description}. Parameters: {', '.join(param_details)}")
            else:
                tools_text.append(f"- {name}: {description}")

        return "\n".join(tools_text)

    def build_boost_message(
        self,
        user_request: str,
        tools: List[Dict[str, Any]],
        loop_count: int = 0,
        previous_attempts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Build message for boost model with tools embedded in content."""
        custom_template = getattr(self.config, "boost_wrapper_template", None)
        template = custom_template if isinstance(custom_template, str) and custom_template.strip() else self._get_default_wrapper_template()

        tools_text = self._format_tools_for_message(tools)

        previous_attempts_text = ""
        if previous_attempts:
            previous_attempts_text = "\n".join([f"- {attempt}" for attempt in previous_attempts])

        message_content = template.format(
            loop_count=loop_count,
            previous_attempts=previous_attempts_text or "None",
            user_request=user_request,
            tools_text=tools_text,
        )

        return {
            "model": self.config.boost_model,
            "messages": [
                {
                    "role": "user",
                    "content": message_content,
                }
            ],
            "temperature": 0.7,
            "max_tokens": 4096,
        }

    async def get_boost_guidance(
        self,
        user_request: str,
        tools: List[Dict[str, Any]],
        loop_count: int = 0,
        previous_attempts: Optional[List[str]] = None,
    ) -> Tuple[str, str, str]:
        """
        Get guidance from boost model.

        Returns:
            Tuple of (response_type, analysis, guidance_or_summary)
        """
        cache_key = self._build_cache_key(user_request, tools, loop_count, previous_attempts)
        cached_entry = await self._get_cached_response(cache_key)
        if cached_entry:
            logger.debug("Using cached boost response for identical input.")
            return cached_entry.response_type, cached_entry.analysis, cached_entry.payload

        message = self.build_boost_message(user_request, tools, loop_count, previous_attempts)
        response = await self.call_boost_model(message)

        sections = self._parse_sections(response)
        response_type, analysis, payload = self._classify_sections(sections)

        await self._store_cached_response(
            cache_key,
            CacheEntry(
                response_type=response_type,
                analysis=analysis or "",
                payload=payload or "",
                raw_response=response,
                created_at=monotonic(),
            ),
        )

        return response_type, analysis or "", payload or ""

    def _extract_section(self, text: str, section_name: str) -> Optional[str]:
        """Extract a section from the boost model response using cached parsing."""
        section_key = section_name.rstrip(":").upper()
        sections = self._parse_sections(text)
        return sections.get(section_key)

    async def _get_cached_response(self, cache_key: str) -> Optional[CacheEntry]:
        """Return cached response if available and not expired."""
        if not cache_key:
            return None

        lock = await self._get_cache_lock()
        async with lock:
            cached = self._response_cache.get(cache_key)
            if not cached:
                return None

            if monotonic() - cached.created_at > self._response_cache_ttl:
                self._response_cache.pop(cache_key, None)
                return None

            self._response_cache.move_to_end(cache_key)
            return cached

    async def _store_cached_response(self, cache_key: str, entry: CacheEntry) -> None:
        """Store a response in cache, enforcing TTL and size limits."""
        if not cache_key:
            return

        lock = await self._get_cache_lock()
        async with lock:
            self._response_cache[cache_key] = entry
            self._response_cache.move_to_end(cache_key)

            while len(self._response_cache) > self._response_cache_limit:
                self._response_cache.popitem(last=False)

    async def _get_cache_lock(self) -> asyncio.Lock:
        """Ensure cache lock is created within an event loop context."""
        if self._response_cache_lock is None:
            self._response_cache_lock = asyncio.Lock()
        return self._response_cache_lock

    def _build_cache_key(
        self,
        user_request: str,
        tools: List[Dict[str, Any]],
        loop_count: int,
        previous_attempts: Optional[List[str]],
    ) -> str:
        """Create a stable cache key from the request context."""
        try:
            tools_repr = json.dumps(tools or [], sort_keys=True, default=str)
        except TypeError:
            tools_repr = repr(tools)

        attempts_repr = json.dumps(previous_attempts or [], sort_keys=False, default=str)

        cache_payload = {
            "model": self.config.boost_model,
            "user_request": user_request,
            "tools": tools_repr,
            "loop_count": loop_count,
            "attempts": attempts_repr,
        }

        return json.dumps(cache_payload, sort_keys=True)

    def _classify_sections(self, sections: Dict[str, Optional[str]]) -> Tuple[str, str, str]:
        """Determine response type and associated payloads from parsed sections."""
        summary = sections.get("SUMMARY")
        guidance = sections.get("GUIDANCE")
        analysis = sections.get("ANALYSIS") or ""

        if summary:
            return "SUMMARY", analysis, summary
        if guidance:
            return "GUIDANCE", analysis, guidance
        if analysis:
            return "OTHER", analysis, ""
        return "OTHER", "", ""

    def _parse_sections(self, text: str) -> Dict[str, Optional[str]]:
        """Parse boost response into sections with lightweight caching."""
        cached_sections = self._section_cache.get(text)
        if cached_sections is not None:
            self._section_cache.move_to_end(text)
            return cached_sections

        sections: Dict[str, List[str]] = {
            "SUMMARY": [],
            "ANALYSIS": [],
            "GUIDANCE": [],
        }
        seen_sections: Set[str] = set()
        current: Optional[str] = None

        for line in text.splitlines():
            stripped = line.strip()

            if stripped.startswith("SUMMARY:"):
                if "SUMMARY" in seen_sections:
                    current = None
                    continue
                seen_sections.add("SUMMARY")
                current = "SUMMARY"
                content = stripped[len("SUMMARY:"):].strip()
                if content:
                    sections[current].append(content)
                continue
            if stripped.startswith("ANALYSIS:"):
                if "ANALYSIS" in seen_sections:
                    current = None
                    continue
                seen_sections.add("ANALYSIS")
                current = "ANALYSIS"
                content = stripped[len("ANALYSIS:"):].strip()
                if content:
                    sections[current].append(content)
                continue
            if stripped.startswith("GUIDANCE:"):
                if "GUIDANCE" in seen_sections:
                    current = None
                    continue
                seen_sections.add("GUIDANCE")
                current = "GUIDANCE"
                content = stripped[len("GUIDANCE:"):].strip()
                if content:
                    sections[current].append(content)
                continue

            if stripped.startswith("---"):
                current = None
                continue

            if current:
                sections[current].append(line.rstrip())

        finalized: Dict[str, Optional[str]] = {}
        for key, values in sections.items():
            while values and values[0] == "":
                values.pop(0)
            while values and values[-1] == "":
                values.pop()
            finalized[key] = "\n".join(values) if values else None

        self._section_cache[text] = finalized
        self._section_cache.move_to_end(text)
        while len(self._section_cache) > self._section_cache_limit:
            self._section_cache.popitem(last=False)

        return finalized

    def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure a pooled client exists for this configuration."""
        with self._pool_lock:
            client = self._client_pool.get(self._pool_key)
            if client is None:
                client = httpx.AsyncClient(
                    base_url=self.config.boost_base_url,
                    timeout=self.config.request_timeout,
                    headers={
                        "Authorization": f"Bearer {self.config.boost_api_key}",
                        "Content-Type": "application/json",
                    },
                    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                )
                self._client_pool[self._pool_key] = client
            return client

    async def _get_or_create_client(self) -> httpx.AsyncClient:
        """Return pooled HTTP client for boost API calls."""
        return self.client

    async def call_boost_model(self, message: Dict[str, Any]) -> str:
        """Call the boost model and return the response text."""
        try:
            logger.info("Calling boost model: %s", self.config.boost_model)

            client = await self._get_or_create_client()
            response = await client.post(
                "/chat/completions",
                json=message,
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            logger.info("Boost model response received: %s characters", len(content))
            return content

        except httpx.HTTPError as error:
            logger.error("HTTP error calling boost model: %s", error)
            raise
        except (KeyError, IndexError) as error:
            logger.error("Invalid response format from boost model: %s", error)
            raise
        except Exception as error:
            logger.error("Unexpected error calling boost model: %s", error)
            raise

    async def close(self):
        """Pooled clients persist between requests; provided for compatibility."""
        return

    @classmethod
    async def close_pools(cls):
        """Close all pooled clients, primarily for shutdown hooks or tests."""
        with cls._pool_lock:
            clients = list(cls._client_pool.values())
            cls._client_pool.clear()

        if clients:
            await asyncio.gather(*(client.aclose() for client in clients), return_exceptions=True)
