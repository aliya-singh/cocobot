import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    """Centralized configuration for the AI Knowledge Companion"""
    
    # ==================== API KEYS ====================
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY", "")
    
    # ==================== LLM SETTINGS ====================
    DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "groq")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    
    # Temperature settings for different response modes
    TEMP_CONCISE = 0.3  # More deterministic, focused
    TEMP_DETAILED = 0.7  # More creative, exploratory
    
    # ==================== RAG SETTINGS ====================
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
    
    # ==================== RATE LIMITING ====================
    MAX_QUERIES_PER_DAY = int(os.getenv("MAX_QUERIES_PER_DAY", 50))
    RATE_LIMIT_SECONDS = int(os.getenv("RATE_LIMIT_SECONDS", 2))
    MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", 10))
    
    # ==================== CACHING ====================
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", 24))
    CACHE_DIR = "./data/cache"
    
    # ==================== WEB SEARCH ====================
    ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"
    MAX_WEB_SEARCH_RESULTS = int(os.getenv("MAX_WEB_SEARCH_RESULTS", 3))
    WEB_SEARCH_CONFIDENCE_THRESHOLD = float(os.getenv("WEB_SEARCH_CONFIDENCE_THRESHOLD", 0.5))
    
    # ==================== LOGGING ====================
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "./logs/app.log")
    
    # ==================== PATHS ====================
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    LOGS_DIR = PROJECT_ROOT / "logs"
    UPLOADED_DOCS_DIR = DATA_DIR / "uploaded_documents"
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
        cls.UPLOADED_DOCS_DIR.mkdir(exist_ok=True)
        Path(cls.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)
        Path(cls.CACHE_DIR).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_api_keys(cls):
        """Validate that required API keys are configured"""
        if cls.DEFAULT_LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not configured")
        if cls.DEFAULT_LLM_PROVIDER == "groq" and not cls.GROQ_API_KEY:
            raise ValueError("Groq API key not configured")
        if cls.DEFAULT_LLM_PROVIDER == "gemini" and not cls.GOOGLE_GEMINI_API_KEY:
            raise ValueError("Google Gemini API key not configured")
    
    @classmethod
    def get_available_providers(cls):
        """Get list of available LLM providers based on configured keys"""
        available = []
        if cls.GROQ_API_KEY:
            available.append("groq")
        if cls.OPENAI_API_KEY:
            available.append("openai")
        if cls.GOOGLE_GEMINI_API_KEY:
            available.append("gemini")
        return available
    
    # ==================== UI SETTINGS ====================
    APP_TITLE = "🔧 AI Knowledge Companion for Engineers"
    APP_ICON = "⚙️"
    
    # System prompt template
    SYSTEM_PROMPT_TEMPLATE = """You are an AI Knowledge Companion for Engineers. You help engineers solve technical problems, understand concepts, and find solutions quickly.

Your characteristics:
- Provide accurate, practical engineering advice
- Cite sources when using external information
- Explain complex concepts clearly
- Include code examples when relevant
- Be concise but thorough
- Ask clarifying questions when needed

When you don't know something, acknowledge it and suggest resources."""
    
    # Response mode templates
    RESPONSE_MODE_TEMPLATES = {
        "concise": "Provide a brief, focused answer with key points only. Keep it under 150 words.",
        "detailed": "Provide a comprehensive, in-depth answer with examples, trade-offs, and detailed explanations. Use formatting where appropriate."
    }


# Initialize config on import
Config.ensure_directories()