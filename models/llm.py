import logging
from typing import List, Dict, Optional, Any
import os

logger = logging.getLogger(__name__)


class GroqLLM:
    """Groq provider - Free, fast, unlimited!"""
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", temperature: float = 0.7):
        """Initialize Groq LLM - WITHOUT proxies"""
        self.api_key = api_key
        self.model = "llama-3.3-70b-versatile"
        self.temperature = temperature
        
        try:
            from groq import Groq
            
            # Create client with ONLY api_key, no other parameters
            # This avoids the proxies error
            self.client = Groq(
                api_key=api_key,
                # Remove ALL other parameters that might cause issues
            )
            logger.info(f"✅ Groq initialized: {self.model}")
        except TypeError as e:
            if "proxies" in str(e):
                logger.error("❌ Groq proxies error - trying workaround")
                # Fallback: direct HTTP client
                try:
                    from groq import Groq
                    # Try with minimal params
                    self.client = Groq(api_key=api_key)
                except Exception as e2:
                    logger.error(f"❌ Groq fallback failed: {e2}")
                    raise
            else:
                raise
        except Exception as e:
            logger.error(f"❌ Groq error: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        """Send chat to Groq"""
        try:
            # Build kwargs with only supported parameters
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
            logger.error(f"❌ Groq chat error: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model info"""
        return {
            "provider": "groq",
            "model": self.model,
            "status": "✅"
        }


class OpenAILLM:
    """OpenAI provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", temperature: float = 0.7):
        """Initialize OpenAI"""
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            logger.info(f"✅ OpenAI initialized")
        except Exception as e:
            logger.error(f"❌ OpenAI error: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        """Send chat to OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ OpenAI chat error: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model info"""
        return {
            "provider": "openai",
            "model": self.model,
            "status": "✅"
        }


class GeminiLLM:
    """Google Gemini provider"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", temperature: float = 0.7):
        """Initialize Gemini"""
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
            logger.info(f"✅ Gemini initialized")
        except Exception as e:
            logger.error(f"❌ Gemini error: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        """Send chat to Gemini"""
        try:
            history = []
            user_msg = None
            
            for msg in messages[:-1]:
                history.append({
                    "role": "user" if msg["role"] == "user" else "model",
                    "parts": msg["content"]
                })
            
            if messages:
                user_msg = messages[-1]["content"]
            
            chat = self.client.start_chat(history=history)
            response = chat.send_message(user_msg)
            return response.text
        except Exception as e:
            logger.error(f"❌ Gemini chat error: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model info"""
        return {
            "provider": "gemini",
            "model": self.model,
            "status": "✅"
        }


class LLMFactory:
    """Factory to create LLM instances"""
    
    @staticmethod
    def create_llm(provider: str, api_key: str, temperature: float = 0.7):
        """Create LLM instance"""
        provider = provider.lower().strip()
        
        if provider == "groq":
            return GroqLLM(api_key, "llama-3.3-70b-versatile", temperature)
        elif provider == "openai":
            return OpenAILLM(api_key, "gpt-3.5-turbo", temperature)
        elif provider == "gemini":
            return GeminiLLM(api_key, "gemini-1.5-flash", temperature)
        else:
            raise ValueError(f"Unknown provider: {provider}")