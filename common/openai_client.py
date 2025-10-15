import json
import os
from typing import Optional, Dict, Any
from openai import OpenAI


class OpenAIClient:
    """
    OpenAI-compatible API client for LLM inference.
    Supports configurable base URL and model parameters.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 60
    ):
        """
        Initialize OpenAI-compatible client.

        Args:
            api_key: OpenAI API key. If None, will try to get from environment.
            base_url: Base URL for OpenAI-compatible API. If None, uses default OpenAI URL.
            model: Model name to use.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "sk-no-key-required")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        format: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response using OpenAI-compatible API.

        Args:
            prompt: Input prompt.
            model: Model name. If None, uses default model.
            temperature: Sampling temperature. If None, uses default temperature.
            max_tokens: Maximum tokens. If None, uses default max_tokens.
            format: Response format. If 'json', sets response_format to {'type': 'json_object'}.
            **kwargs: Additional parameters for the API call.

        Returns:
            Dictionary containing the response.
        """
        # Use defaults if not provided
        model = model or self.model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        # Prepare request parameters
        request_params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add format if specified
        if format == 'json':
            request_params["response_format"] = {"type": "json_object"}

        # Add any additional parameters
        request_params.update(kwargs)

        try:
            response = self.client.chat.completions.create(**request_params)
            return {
                'response': response.choices[0].message.content,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                } if response.usage else None
            }
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")

    def __call__(self, *args, **kwargs):
        """Make the client callable to maintain compatibility with ollama.generate."""
        return self.generate(*args, **kwargs)


def create_openai_client(config) -> OpenAIClient:
    """
    Create OpenAI client from configuration.

    Args:
        config: Configuration object with llm, openai_api_key, openai_base_url attributes.

    Returns:
        OpenAIClient instance.
    """
    # Get configuration values
    model = getattr(config, 'llm', 'gpt-3.5-turbo')
    api_key = getattr(config, 'openai_api_key', None)
    base_url = getattr(config, 'openai_base_url', None)
    temperature = getattr(config, 'temperature', 0.0)
    max_tokens = getattr(config, 'max_tokens', 4096)
    timeout = getattr(config, 'timeout', 60)

    return OpenAIClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )


# Global client instance (for backward compatibility)
_global_client: Optional[OpenAIClient] = None


def initialize_client(config):
    """Initialize global client instance."""
    global _global_client
    _global_client = create_openai_client(config)


def generate(*args, **kwargs):
    """
    Global generate function for backward compatibility with ollama.generate.
    Must call initialize_client first.
    """
    if _global_client is None:
        raise RuntimeError("OpenAI client not initialized. Call initialize_client(config) first.")
    return _global_client.generate(*args, **kwargs)