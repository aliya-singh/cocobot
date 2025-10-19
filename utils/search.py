"""Document search and retrieval utilities"""
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class DocumentSearch:
    """Search documents in knowledge base"""
    
    @staticmethod
    def keyword_search(query: str, documents: Dict[str, str], limit: int = 5) -> str:
        """Simple keyword search in documents"""
        if not documents:
            return ""
        
        query_words = query.lower().split()
        results = []
        
        for doc_name, doc_text in documents.items():
            doc_lower = doc_text.lower()
            
            # Check if query words are in document
            matches = sum(1 for word in query_words if word in doc_lower)
            
            if matches > 0:
                # Extract relevant lines
                lines = doc_text.split('\n')
                relevant_lines = [
                    line for line in lines 
                    if any(word in line.lower() for word in query_words)
                ]
                
                if relevant_lines:
                    snippet = ' '.join(relevant_lines[:3])[:300]
                    results.append((doc_name, snippet, matches))
        
        # Sort by relevance
        results.sort(key=lambda x: x[2], reverse=True)
        
        # Format output
        if results:
            context = "KNOWLEDGE BASE RESULTS:\n"
            for doc_name, snippet, _ in results[:limit]:
                context += f"\n[{doc_name}]\n{snippet}\n"
            return context
        
        return ""
    
    @staticmethod
    def build_rag_context(query: str, documents: Dict[str, str]) -> str:
        """Build RAG context for prompt"""
        if not documents:
            return ""
        
        context = DocumentSearch.keyword_search(query, documents)
        
        if not context:
            context = "No relevant documents found in knowledge base.\n"
        
        return context