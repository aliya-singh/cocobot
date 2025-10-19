"""Groq API client for direct REST calls"""
import requests
import logging
from typing import List, Dict
from config.config import Config

logger = logging.getLogger(__name__)

class GroqClient:
    """Direct Groq API client using REST"""
    
    def __init__(self, api_key: str = None):
        """Initialize Groq client"""
        self.api_key = api_key or Config.GROQ_API_KEY
        self.model = Config.GROQ_MODEL
        self.url = Config.GROQ_API_URL
        self.timeout = Config.REQUEST_TIMEOUT
        
        if not self.api_key:
            raise ValueError("Groq API key not provided")
    
    def chat(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = None) -> str:
        """Send chat request to Groq"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens or Config.MAX_TOKENS
            }
            
            response = requests.post(
                self.url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            else:
                logger.error(f"Groq API error: {response.status_code}")
                return f"Error: {response.status_code} - {response.text}"
        
        except Exception as e:
            logger.error(f"Groq request failed: {e}")
            raise
    
    def is_healthy(self) -> bool:
        """Check if Groq API is accessible"""
        try:
            test_messages = [{"role": "user", "content": "ping"}]
            self.chat(test_messages)
            return True
        except:
            return False