"""
Base LLM interface
"""

from abc import ABC, abstractmethod


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces"""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        pass

    @abstractmethod
    async def generate_with_context(
        self, system_message: str, messages: list[dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        pass
