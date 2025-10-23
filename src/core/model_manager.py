from src.core.config import config

class ModelManager:
    def __init__(self, config):
        self.config = config

    def map_claude_model_to_openai(self, claude_model: str) -> str:
        """Map Claude model names to OpenAI model names based on BIG/SMALL pattern"""
        # If it's already an OpenAI model, return as-is
        if claude_model.startswith("gpt-") or claude_model.startswith("o1-"):
            return claude_model

        # If it's other supported models (ARK/Doubao/DeepSeek), return as-is
        if (claude_model.startswith("ep-") or claude_model.startswith("doubao-") or
            claude_model.startswith("deepseek-")):
            return claude_model

        # Map based on model naming patterns
        model_lower = claude_model.lower()
        if 'haiku' in model_lower:
            return self.config.small_model
        elif 'sonnet' in model_lower:
            return self.config.middle_model
        elif 'opus' in model_lower:
            return self.config.big_model
        else:
            # Default to big model for unknown models
            return self.config.big_model

    def get_model_tier(self, claude_model: str) -> str:
        """Determine which tier a Claude model belongs to"""
        model_lower = claude_model.lower()

        # Check mapping patterns
        if 'haiku' in model_lower:
            return "SMALL_MODEL"
        elif 'sonnet' in model_lower:
            return "MIDDLE_MODEL"
        elif 'opus' in model_lower:
            return "BIG_MODEL"
        else:
            # For OpenAI models, determine by the actual model name
            if claude_model == self.config.small_model:
                return "SMALL_MODEL"
            elif claude_model == self.config.middle_model:
                return "MIDDLE_MODEL"
            elif claude_model == self.config.big_model:
                return "BIG_MODEL"
            else:
                # Default to BIG_MODEL for unknown models
                return "BIG_MODEL"

    def has_tools(self, request) -> bool:
        """Check if the request includes tools"""
        # Check for tools in various formats
        if hasattr(request, 'tools') and request.tools:
            return True

        # Check if it's a converted OpenAI request with tools
        if isinstance(request, dict) and request.get('tools'):
            return True

        return False

model_manager = ModelManager(config)