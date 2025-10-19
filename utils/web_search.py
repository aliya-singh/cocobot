import logging
import requests
from typing import List, Dict, Any, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)


class WebSearchEngine:
    """
    Free web search using DuckDuckGo API
    No API key required!
    """
    
    BASE_URL = "https://api.duckduckgo.com"
    TIMEOUT = 5
    
    @staticmethod
    def search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web using DuckDuckGo
        
        Args:
            query: Search query
            max_results: Maximum results to return
        
        Returns:
            List of search results
        """
        try:
            # DuckDuckGo API endpoint
            params = {
                "q": query,
                "format": "json",
                "no_redirect": 1,
                "no_html": 1,
                "t": "engineering-companion"  # User agent identifier
            }
            
            response = requests.get(
                WebSearchEngine.BASE_URL,
                params=params,
                timeout=WebSearchEngine.TIMEOUT
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Parse results
            results = []
            
            # Abstract results
            if data.get("AbstractText"):
                results.append({
                    "title": data.get("AbstractTitle", "Direct Answer"),
                    "snippet": data.get("AbstractText", ""),
                    "url": data.get("AbstractURL", ""),
                    "source": "direct_answer"
                })
            
            # Topic results
            for result in data.get("Results", [])[:max_results]:
                results.append({
                    "title": result.get("Text", ""),
                    "snippet": result.get("FirstURL", ""),
                    "url": result.get("FirstURL", ""),
                    "source": "duckduckgo"
                })
            
            # Related results
            for result in data.get("RelatedTopics", [])[:max_results - len(results)]:
                if "Text" in result:
                    results.append({
                        "title": result.get("Text", ""),
                        "snippet": result.get("FirstURL", ""),
                        "url": result.get("FirstURL", ""),
                        "source": "related"
                    })
            
            logger.info(f"Web search returned {len(results)} results for: {query}")
            return results[:max_results]
        
        except requests.Timeout:
            logger.warning(f"Web search timeout for query: {query}")
            return []
        except requests.RequestException as e:
            logger.error(f"Web search error: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in web search: {e}")
            return []
    
    @staticmethod
    def search_engineering(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Engineering-focused search (adds relevant keywords)
        
        Args:
            query: Search query
            max_results: Maximum results
        
        Returns:
            List of search results
        """
        # Add engineering-relevant terms to boost results
        engineering_query = f"{query} (documentation OR tutorial OR guide OR best practices)"
        
        return WebSearchEngine.search(engineering_query, max_results)


class GoogleWebSearch:
    """
    Alternative: Google Search (if you have API key)
    Uses Custom Search Engine (CSE)
    """
    
    def __init__(self, api_key: str, search_engine_id: str):
        """
        Initialize Google Search
        
        Args:
            api_key: Google API key
            search_engine_id: Custom Search Engine ID
        """
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search using Google Custom Search
        
        Args:
            query: Search query
            max_results: Maximum results
        
        Returns:
            List of search results
        """
        try:
            params = {
                "q": query,
                "key": self.api_key,
                "cx": self.search_engine_id,
                "num": min(max_results, 10)
            }
            
            response = requests.get(self.base_url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            results = []
            for item in data.get("items", []):
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("link", ""),
                    "source": "google"
                })
            
            logger.info(f"Google search returned {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Google search error: {e}")
            return []


class SearchResult:
    """Enhanced search result with formatting"""
    
    def __init__(self, title: str, snippet: str, url: str, source: str = "unknown"):
        """Initialize search result"""
        self.title = title
        self.snippet = snippet
        self.url = url
        self.source = source
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "snippet": self.snippet,
            "url": self.url,
            "source": self.source
        }
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.title}\n{self.snippet}\n{self.url}"


class SearchManager:
    """
    Unified search manager
    Routes to best available search method
    """
    
    def __init__(self, use_duckduckgo: bool = True, google_key: str = "", google_cse_id: str = ""):
        """
        Initialize search manager
        
        Args:
            use_duckduckgo: Use DuckDuckGo (free, no key)
            google_key: Google API key (optional)
            google_cse_id: Google CSE ID (optional)
        """
        self.use_duckduckgo = use_duckduckgo
        self.google_search = None
        
        if google_key and google_cse_id:
            self.google_search = GoogleWebSearch(google_key, google_cse_id)
    
    def search(self, query: str, max_results: int = 5, method: str = "auto") -> List[Dict[str, Any]]:
        """
        Perform search
        
        Args:
            query: Search query
            max_results: Maximum results
            method: Search method (auto, duckduckgo, google)
        
        Returns:
            List of search results
        """
        if method == "auto":
            # Use DuckDuckGo by default (always works)
            method = "duckduckgo" if self.use_duckduckgo else "google"
        
        if method == "duckduckgo":
            return WebSearchEngine.search(query, max_results)
        elif method == "google" and self.google_search:
            return self.google_search.search(query, max_results)
        else:
            logger.warning(f"Search method not available: {method}")
            return []
    
    def search_engineering(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Engineering-focused search"""
        return WebSearchEngine.search_engineering(query, max_results)


# Global search instance
_search_manager = None


def get_search_manager() -> SearchManager:
    """Get or create search manager (singleton)"""
    global _search_manager
    
    if _search_manager is None:
        _search_manager = SearchManager(use_duckduckgo=True)
    
    return _search_manager


# Convenience functions
def search_web(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Quick web search"""
    manager = get_search_manager()
    return manager.search(query, max_results)


def search_engineering(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Quick engineering search"""
    manager = get_search_manager()
    return manager.search_engineering(query, max_results)