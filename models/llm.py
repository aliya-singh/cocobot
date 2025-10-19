import logging
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
import json

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Base class for LLM providers"""
    
    def __init__(self, api_key: str, model: str, temperature: float = 0.7):
        """
        Initialize LLM
        
        Args:
            api_key: API key for the provider
            model: Model name
            temperature: Temperature for response generation (0-1)
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        """
        Send chat messages and get response
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens in response
        
        Returns:
            Response text
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        pass


class GroqLLM(BaseLLM):
    """Groq provider - Free, fast, unlimited!"""
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", temperature: float = 0.7):
        """Initialize Groq LLM"""
        super().__init__(api_key, model, temperature)
        
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
            logger.info(f"Groq client initialized with model: {model}")
        except ImportError:
            raise ImportError("groq package not installed. Install with: pip install groq")
        except Exception as e:
            logger.error(f"Failed to initialize Groq: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        """
        Send chat request to Groq
        
        Args:
            messages: Chat messages
            max_tokens: Max response tokens
        
        Returns:
            Response text
        """
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            
            response = self.client.chat.completions.create(**kwargs)
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Groq model info"""
        return {
            "provider": "groq",
            "model": self.model,
            "temperature": self.temperature,
            "cost_per_1m_tokens": "$0 (FREE!)",
            "context_window": "128K tokens" if "70b" in self.model else "32K tokens",
            "speed": "Very Fast (50+ req/sec)"
        }


class OpenAILLM(BaseLLM):
    """OpenAI provider - Premium quality"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", temperature: float = 0.7):
        """Initialize OpenAI LLM"""
        super().__init__(api_key, model, temperature)
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            logger.info(f"OpenAI client initialized with model: {model}")
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        """
        Send chat request to OpenAI
        
        Args:
            messages: Chat messages
            max_tokens: Max response tokens
        
        Returns:
            Response text
        """
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            
            response = self.client.chat.completions.create(**kwargs)
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model info"""
        costs = {
            "gpt-3.5-turbo": {"input": "$0.50", "output": "$1.50"},
            "gpt-4": {"input": "$30", "output": "$60"},
            "gpt-4o-mini": {"input": "$0.15", "output": "$0.60"},
        }
        
        cost_info = costs.get(self.model, {"input": "Unknown", "output": "Unknown"})
        
        return {
            "provider": "openai",
            "model": self.model,
            "temperature": self.temperature,
            "cost_per_1m_tokens": cost_info,
            "context_window": "4K-128K tokens depending on model",
            "speed": "Standard"
        }


class GeminiLLM(BaseLLM):
    """Google Gemini provider - Balanced quality/cost"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", temperature: float = 0.7):
        """Initialize Gemini LLM"""
        super().__init__(api_key, model, temperature)
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
            logger.info(f"Gemini client initialized with model: {model}")
        except ImportError:
            raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        """
        Send chat request to Gemini
        
        Args:
            messages: Chat messages
            max_tokens: Max response tokens
        
        Returns:
            Response text
        """
        try:
            # Gemini expects conversation history differently
            # Convert to Gemini format
            history = []
            user_message = None
            
            for msg in messages[:-1]:  # All but last are history
                history.append({
                    "role": "user" if msg["role"] == "user" else "model",
                    "parts": msg["content"]
                })
            
            # Last message is current query
            if messages:
                user_message = messages[-1]["content"]
            
            # Start chat session
            chat = self.client.start_chat(history=history)
            
            kwargs = {"generation_config": {"temperature": self.temperature}}
            if max_tokens:
                kwargs["generation_config"]["max_output_tokens"] = max_tokens
            
            response = chat.send_message(user_message, **kwargs)
            
            return response.text
        
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Gemini model info"""
        return {
            "provider": "gemini",
            "model": self.model,
            "temperature": self.temperature,
            "cost_per_1m_tokens": {"input": "$0.075", "output": "$0.30"},
            "context_window": "1M tokens",
            "speed": "Fast"
        }


class LLMFactory:
    """Factory to create LLM instances based on provider"""
    
    @staticmethod
    def create_llm(provider: str, api_key: str, model: str = None, temperature: float = 0.7) -> BaseLLM:
        """
        Create LLM instance
        
        Args:
            provider: Provider name (groq, openai, gemini)
            api_key: API key
            model: Model name (uses default if None)
            temperature: Temperature parameter
        
        Returns:
            LLM instance
        """
        provider = provider.lower().strip()
        
        if provider == "groq":
            from config.config import Config
            model = model or Config.GROQ_MODEL
            return GroqLLM(api_key, model, temperature)
        
        elif provider == "openai":
            from config.config import Config
            model = model or Config.OPENAI_MODEL
            return OpenAILLM(api_key, model, temperature)
        
        elif provider == "gemini":
            from config.config import Config
            model = model or Config.GEMINI_MODEL
            return GeminiLLM(api_key, model, temperature)
        
        else:
            raise ValueError(f"Unknown provider: {provider}. Use: groq, openai, or gemini")
    
    @staticmethod
    def get_best_available_provider(groq_key: str = "", openai_key: str = "", gemini_key: str = "") -> str:
        """
        Get best available provider based on API keys (Groq > Gemini > OpenAI in terms of cost)
        
        Args:
            groq_key: Groq API key
            openai_key: OpenAI API key
            gemini_key: Gemini API key
        
        Returns:
            Provider name
        """
        # Priority: Groq (free) > Gemini (cheap) > OpenAI (expensive)
        if groq_key:
            return "groq"
        elif gemini_key:
            return "gemini"
        elif openai_key:
            return "openai"
        else:
            raise ValueError("No API keys provided!")


# Convenience functions
def create_llm(provider: str, api_key: str, temperature: float = 0.7) -> BaseLLM:
    """Quick LLM creation"""
    return LLMFactory.create_llm(provider, api_key, temperature=temperature)