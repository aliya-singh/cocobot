"""Configuration settings for AI Knowledge Companion"""
import os
from pathlib import Path

class Config:
    """Application configuration"""
    
    # API Settings
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "llama-3.3-70b-versatile"
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    
    # RAG Settings
    MAX_DOCUMENT_SIZE = 3000  # Characters
    MAX_DOCUMENTS = 20
    SEARCH_RESULTS_LIMIT = 5
    
    # Response Settings
    MAX_TOKENS = 2000
    DEFAULT_TEMPERATURE_CONCISE = 0.3
    DEFAULT_TEMPERATURE_DETAILED = 0.7
    
    # File Upload Settings
    ALLOWED_EXTENSIONS = ['txt', 'md', 'pdf']
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Request Settings
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment")
        return True