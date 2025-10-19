import json
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Simple file-based cache to avoid duplicate API calls
    Reduces costs by caching responses and embeddings
    """
    
    def __init__(self, cache_dir: str = "./data/cache"):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def _generate_key(query: str, prefix: str = "") -> str:
        """
        Generate a consistent cache key from query
        
        Args:
            query: Query string
            prefix: Optional prefix (e.g., "llm_", "embedding_")
        
        Returns:
            Hash-based cache key
        """
        # Normalize query (lowercase, strip whitespace)
        normalized = query.lower().strip()
        
        # Create hash
        hash_obj = hashlib.md5(normalized.encode())
        hash_key = hash_obj.hexdigest()
        
        return f"{prefix}{hash_key}"
    
    def get(self, key: str, ttl_hours: int = 24) -> Optional[Any]:
        """
        Retrieve value from cache
        
        Args:
            key: Cache key
            ttl_hours: Time-to-live in hours (default 24h)
        
        Returns:
            Cached value or None if expired/not found
        """
        cache_file = self.cache_dir / f"{key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check expiration
            created_at = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - created_at > timedelta(hours=ttl_hours):
                logger.info(f"Cache expired: {key}")
                self.delete(key)
                return None
            
            logger.info(f"Cache hit: {key}")
            return cache_data['value']
        
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def set(self, key: str, value: Any) -> bool:
        """
        Store value in cache
        
        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
        
        Returns:
            True if successful
        """
        cache_file = self.cache_dir / f"{key}.json"
        
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'value': value
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            logger.info(f"Cache set: {key}")
            return True
        
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete a cache entry"""
        cache_file = self.cache_dir / f"{key}.json"
        
        try:
            if cache_file.exists():
                cache_file.unlink()
            return True
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
            return False
    
    def clear_expired(self, ttl_hours: int = 24):
        """Clear all expired cache entries"""
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                created_at = datetime.fromisoformat(cache_data['timestamp'])
                if datetime.now() - created_at > timedelta(hours=ttl_hours):
                    cache_file.unlink()
                    count += 1
            except Exception as e:
                logger.warning(f"Error clearing cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {count} expired cache entries")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'total_entries': len(cache_files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_dir': str(self.cache_dir)
        }


class QueryCache:
    """
    Specific cache for LLM query responses
    Prevents duplicate API calls for same query
    """
    
    def __init__(self, cache_manager: CacheManager):
        """
        Initialize query cache
        
        Args:
            cache_manager: CacheManager instance
        """
        self.cache_manager = cache_manager
        self.prefix = "llm_query_"
    
    def get_response(self, query: str, provider: str = "", ttl_hours: int = 24) -> Optional[str]:
        """
        Get cached response for query
        
        Args:
            query: User query
            provider: LLM provider (optional)
            ttl_hours: Cache TTL
        
        Returns:
            Cached response or None
        """
        key = self.cache_manager._generate_key(f"{provider}_{query}", self.prefix)
        return self.cache_manager.get(key, ttl_hours)
    
    def cache_response(self, query: str, response: str, provider: str = "") -> bool:
        """
        Cache a response
        
        Args:
            query: User query
            response: LLM response
            provider: LLM provider
        
        Returns:
            True if successful
        """
        key = self.cache_manager._generate_key(f"{provider}_{query}", self.prefix)
        return self.cache_manager.set(key, response)


class EmbeddingCache:
    """
    Specific cache for document embeddings
    Prevents re-embedding same documents
    """
    
    def __init__(self, cache_manager: CacheManager):
        """
        Initialize embedding cache
        
        Args:
            cache_manager: CacheManager instance
        """
        self.cache_manager = cache_manager
        self.prefix = "embedding_"
    
    def get_embedding(self, text: str, ttl_hours: int = 720) -> Optional[list]:
        """
        Get cached embedding
        
        Args:
            text: Text that was embedded
            ttl_hours: Cache TTL (default 30 days)
        
        Returns:
            Cached embedding as list or None
        """
        key = self.cache_manager._generate_key(text, self.prefix)
        cached = self.cache_manager.get(key, ttl_hours)
        return cached
    
    def cache_embedding(self, text: str, embedding: list) -> bool:
        """
        Cache an embedding
        
        Args:
            text: Text that was embedded
            embedding: Embedding vector (as list)
        
        Returns:
            True if successful
        """
        key = self.cache_manager._generate_key(text, self.prefix)
        return self.cache_manager.set(key, embedding)


# Global cache instances
_cache_manager = None


def get_cache_manager(cache_dir: str = "./data/cache") -> CacheManager:
    """Get or create cache manager (singleton)"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(cache_dir)
    return _cache_manager