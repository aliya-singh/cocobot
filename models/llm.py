import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class GroqLLM:
    """Groq provider - Free, fast, unlimited!"""
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", temperature: float = 0.7):
        """Initialize Groq LLM"""
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        
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
        """Send chat request to Groq"""
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
        """Get model information"""
        return {
            "provider": "groq",
            "model": self.model,
            "temperature": self.temperature,
            "status": "✅ Working"
        }


class OpenAILLM:
    """OpenAI provider - Premium quality"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", temperature: float = 0.7):
        """Initialize OpenAI LLM"""
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        
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
        """Send chat request to OpenAI"""
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
        """Get model information"""
        return {
            "provider": "openai",
            "model": self.model,
            "temperature": self.temperature,
            "status": "✅ Working"
        }


class GeminiLLM:
    """Google Gemini provider - Balanced quality/cost"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", temperature: float = 0.7):
        """Initialize Gemini LLM"""
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        
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
        """Send chat request to Gemini"""
        try:
            # Convert messages to Gemini format
            history = []
            user_message = None
            
            for msg in messages[:-1]:
                history.append({
                    "role": "user" if msg["role"] == "user" else "model",
                    "parts": msg["content"]
                })
            
            if messages:
                user_message = messages[-1]["content"]
            
            chat = self.client.start_chat(history=history)
            response = chat.send_message(user_message)
            
            return response.text
        
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "provider": "gemini",
            "model": self.model,
            "temperature": self.temperature,
            "status": "✅ Working"
        }


class LLMFactory:
    """Factory to create LLM instances"""
    
    @staticmethod
    def create_llm(provider: str, api_key: str, model: str = None, temperature: float = 0.7):
        """Create LLM instance"""
        from config.config import Config
        
        provider = provider.lower().strip()
        
        if provider == "groq":
            model = model or Config.GROQ_MODEL
            return GroqLLM(api_key, model, temperature)
        
        elif provider == "openai":
            model = model or Config.OPENAI_MODEL
            return OpenAILLM(api_key, model, temperature)
        
        elif provider == "gemini":
            model = model or Config.GEMINI_MODEL
            return GeminiLLM(api_key, model, temperature)
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    @staticmethod
    def get_best_available_provider(groq_key: str = "", openai_key: str = "", gemini_key: str = "") -> str:
        """Get best available provider"""
        if groq_key:
            return "groq"
        elif gemini_key:
            return "gemini"
        elif openai_key:
            return "openai"
        else:
            raise ValueError("No API keys provided!")