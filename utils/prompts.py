"""
Prompt engineering module
Optimized prompts for different response modes and use cases
"""


class PromptTemplates:
    """Collection of system and user prompt templates"""
    
    # ==================== SYSTEM PROMPTS ====================
    
    SYSTEM_CONCISE = """You are an AI Knowledge Companion for Engineers. You help engineers solve technical problems and find solutions quickly.

RESPONSE STYLE: Concise and to the point
- Answer in 2-3 sentences maximum
- Only include the most critical information
- Use bullet points for lists (max 3 items)
- Skip explanations, focus on the answer
- No preamble or lengthy introductions

You are helpful, accurate, and direct."""
    
    SYSTEM_DETAILED = """You are an AI Knowledge Companion for Engineers. You help engineers solve technical problems, understand concepts, and find solutions.

RESPONSE STYLE: Comprehensive and detailed
- Provide thorough explanations with context
- Include examples, code snippets, or diagrams when relevant
- Discuss trade-offs and alternatives
- Explain the "why" not just the "what"
- Use formatting (headers, lists, code blocks) for clarity
- Include warnings or edge cases if applicable

You are helpful, thorough, and educational."""
    
    SYSTEM_WITH_CONTEXT = """You are an AI Knowledge Companion for Engineers. You help engineers solve technical problems using available documentation and knowledge.

IMPORTANT: 
1. Use the provided context/documents to answer questions
2. Cite your sources when using external information
3. If information is not in the context, acknowledge it
4. Be honest about knowledge limitations

You are accurate, helpful, and transparent about sources."""
    
    # ==================== SYSTEM PROMPTS WITH RAG CONTEXT ====================
    
    SYSTEM_RAG_CONCISE = """You are an AI Knowledge Companion for Engineers with access to documentation.

INSTRUCTIONS:
1. Answer using the provided documentation snippets
2. Keep responses concise (2-3 sentences)
3. Cite the source document for each fact
4. Format citations as [Source: Document Name]
5. If the answer isn't in the docs, say so

Be direct, accurate, and cite your sources."""
    
    SYSTEM_RAG_DETAILED = """You are an AI Knowledge Companion for Engineers with access to documentation.

INSTRUCTIONS:
1. Answer comprehensively using the provided documentation
2. Include examples and explanations
3. Cite sources for all information: [Source: Document Name]
4. Discuss related concepts from the docs
5. Highlight important warnings or edge cases
6. If something is unclear, ask for clarification

Be thorough, cite sources, and provide context."""
    
    # ==================== USER PROMPT TEMPLATES ====================
    
    @staticmethod
    def build_chat_message(user_input: str, mode: str = "concise", context: str = "") -> dict:
        """
        Build a chat message with optional context
        
        Args:
            user_input: The user's question/query
            mode: Response mode (concise or detailed)
            context: Optional context (from RAG or web search)
        
        Returns:
            Message dict for chat API
        """
        if context:
            # Add context to the message
            message_content = f"""Here is relevant context:

{context}

---

Question: {user_input}

Please answer based on the above context."""
        else:
            message_content = user_input
        
        return {
            "role": "user",
            "content": message_content
        }
    
    @staticmethod
    def build_system_prompt(mode: str = "concise", include_rag: bool = False) -> str:
        """
        Build system prompt based on mode and context
        
        Args:
            mode: Response mode (concise or detailed)
            include_rag: Whether this is for RAG-augmented response
        
        Returns:
            System prompt string
        """
        if include_rag:
            return PromptTemplates.SYSTEM_RAG_CONCISE if mode == "concise" else PromptTemplates.SYSTEM_RAG_DETAILED
        else:
            return PromptTemplates.SYSTEM_CONCISE if mode == "concise" else PromptTemplates.SYSTEM_DETAILED
    
    @staticmethod
    def format_rag_context(documents: list, max_length: int = 2000) -> str:
        """
        Format retrieved documents for inclusion in prompt
        
        Args:
            documents: List of document dicts with 'content' and optionally 'source'
            max_length: Maximum total context length
        
        Returns:
            Formatted context string
        """
        formatted = []
        current_length = 0
        
        for doc in documents:
            source = doc.get('source', 'Unknown')
            content = doc.get('content', '')
            
            # Create formatted chunk
            chunk = f"[{source}]\n{content}\n"
            chunk_length = len(chunk)
            
            # Check if adding this would exceed limit
            if current_length + chunk_length > max_length:
                # Try to fit partial content
                remaining = max_length - current_length - 50  # -50 for ellipsis
                if remaining > 100:
                    chunk = f"[{source}]\n{content[:remaining]}...\n"
                else:
                    break
            
            formatted.append(chunk)
            current_length += len(chunk)
            
            if current_length >= max_length:
                break
        
        return "".join(formatted)
    
    @staticmethod
    def format_web_search_context(results: list, max_results: int = 3) -> str:
        """
        Format web search results for inclusion in prompt
        
        Args:
            results: List of search result dicts
            max_results: Maximum results to include
        
        Returns:
            Formatted context string
        """
        formatted = []
        
        for i, result in enumerate(results[:max_results], 1):
            title = result.get('title', 'No title')
            snippet = result.get('snippet', '')
            url = result.get('url', '')
            
            chunk = f"{i}. {title}\n{snippet}\nSource: {url}\n"
            formatted.append(chunk)
        
        return "\n".join(formatted)
    
    @staticmethod
    def build_rag_prompt(query: str, context: str, mode: str = "concise") -> dict:
        """
        Build complete RAG-augmented prompt
        
        Args:
            query: User query
            context: Retrieved context from documents
            mode: Response mode
        
        Returns:
            System prompt and user message as tuple
        """
        system_prompt = PromptTemplates.build_system_prompt(mode, include_rag=True)
        
        user_message = f"""Context from documentation:

{context}

---

Question: {query}

Please provide a helpful answer based on the above context. Cite the source documents in your response."""
        
        return {
            "system": system_prompt,
            "user": {
                "role": "user",
                "content": user_message
            }
        }


# Quick access functions
def get_system_prompt(mode: str = "concise", rag: bool = False) -> str:
    """Quick access to system prompt"""
    return PromptTemplates.build_system_prompt(mode, rag)


def format_rag_context(documents: list) -> str:
    """Quick access to context formatting"""
    return PromptTemplates.format_rag_context(documents)


def format_web_context(results: list) -> str:
    """Quick access to web search formatting"""
    return PromptTemplates.format_web_search_context(results)