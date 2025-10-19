import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class TokenCounter:
    """
    Estimate tokens before making API calls
    Helps avoid unexpected API costs
    Uses approximation: ~4 chars = 1 token (GPT-style)
    """
    
    # Token estimation ratios for different models
    TOKEN_RATIOS = {
        "gpt-3.5-turbo": 4,      # ~4 chars per token
        "gpt-4": 4,               # ~4 chars per token
        "gpt-4o-mini": 4,         # ~4 chars per token
        "llama-3.3-70b-versatile": 4,  # ~4 chars per token (Groq)
        "llama2-70b": 4,          # ~4 chars per token
        "gemini-1.5-flash": 4,    # ~4 chars per token
    }
    
    # Approximate token costs (USD per 1M tokens)
    TOKEN_COSTS = {
        "gpt-3.5-turbo": {
            "input": 0.50,
            "output": 1.50
        },
        "gpt-4": {
            "input": 30.0,
            "output": 60.0
        },
        "gpt-4o-mini": {
            "input": 0.15,
            "output": 0.60
        },
        "llama-3.3-70b-versatile": {  # Groq - essentially free
            "input": 0.00,
            "output": 0.00
        },
        "llama2-70b": {  # Groq - essentially free
            "input": 0.00,
            "output": 0.00
        },
        "gemini-1.5-flash": {
            "input": 0.075,
            "output": 0.30
        },
    }
    
    @staticmethod
    def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Estimate number of tokens in text
        
        Args:
            text: Text to estimate
            model: Model name for estimation ratio
        
        Returns:
            Estimated token count
        """
        ratio = TokenCounter.TOKEN_RATIOS.get(model, 4)
        estimated_tokens = len(text) / ratio
        return max(1, int(estimated_tokens))
    
    @staticmethod
    def count_message_tokens(messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo") -> int:
        """
        Estimate tokens in chat messages
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name
        
        Returns:
            Estimated total token count (includes overhead)
        """
        total = 0
        
        # ~4 tokens per message for role/metadata overhead
        total += len(messages) * 4
        
        # Count content tokens
        for msg in messages:
            content = msg.get('content', '')
            total += TokenCounter.estimate_tokens(content, model)
        
        return total
    
    @staticmethod
    def estimate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-3.5-turbo") -> float:
        """
        Estimate cost of API call
        
        Args:
            input_tokens: Estimated input tokens
            output_tokens: Estimated output tokens
            model: Model name
        
        Returns:
            Estimated cost in USD
        """
        costs = TokenCounter.TOKEN_COSTS.get(model, {"input": 0, "output": 0})
        
        input_cost = (input_tokens / 1_000_000) * costs["input"]
        output_cost = (output_tokens / 1_000_000) * costs["output"]
        
        return input_cost + output_cost
    
    @staticmethod
    def estimate_response_tokens(prompt_tokens: int, model: str = "gpt-3.5-turbo") -> int:
        """
        Estimate tokens in response based on prompt
        Rough heuristic: response ~= prompt length
        
        Args:
            prompt_tokens: Number of prompt tokens
            model: Model name
        
        Returns:
            Estimated response tokens
        """
        # For most models, response is similar length to prompt
        # This is very rough - actual responses vary
        return int(prompt_tokens * 1.5)
    
    @staticmethod
    def should_proceed(query: str, model: str, max_cost: float = 0.01) -> Dict[str, Any]:
        """
        Check if query should proceed based on cost estimation
        
        Args:
            query: User query
            model: Model to use
            max_cost: Maximum acceptable cost in USD
        
        Returns:
            Dict with 'should_proceed' boolean and cost breakdown
        """
        input_tokens = TokenCounter.estimate_tokens(query, model)
        output_tokens = TokenCounter.estimate_response_tokens(input_tokens, model)
        
        estimated_cost = TokenCounter.estimate_cost(
            input_tokens, 
            output_tokens, 
            model
        )
        
        should_proceed = estimated_cost <= max_cost
        
        return {
            "should_proceed": should_proceed,
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": output_tokens,
            "estimated_cost": round(estimated_cost, 6),
            "max_cost": max_cost,
            "model": model,
            "warning": "Cost exceeds limit!" if not should_proceed else None
        }
    
    @staticmethod
    def log_usage(model: str, input_tokens: int, output_tokens: int):
        """Log token usage for monitoring"""
        cost = TokenCounter.estimate_cost(input_tokens, output_tokens, model)
        logger.info(
            f"Model: {model} | Input: {input_tokens} | Output: {output_tokens} | "
            f"Cost: ${cost:.6f}"
        )
    
    @staticmethod
    def get_model_info(model: str) -> Dict[str, Any]:
        """Get detailed info about a model"""
        return {
            "model": model,
            "token_ratio": TokenCounter.TOKEN_RATIOS.get(model, "unknown"),
            "costs": TokenCounter.TOKEN_COSTS.get(model, {"input": 0, "output": 0}),
            "is_free": TokenCounter.TOKEN_COSTS.get(model, {}).get("input", 0) == 0
        }


# Quick utility functions
def estimate_tokens(text: str) -> int:
    """Quick estimate of tokens (default model)"""
    return TokenCounter.estimate_tokens(text)


def estimate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-3.5-turbo") -> float:
    """Quick cost estimation"""
    return TokenCounter.estimate_cost(input_tokens, output_tokens, model)