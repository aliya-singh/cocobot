import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Simple rate limiter to protect against API quota overuse
    Uses in-memory tracking (resets on app restart)
    """
    
    def __init__(self, max_requests_per_day: int = 50, min_seconds_between_requests: int = 2):
        """
        Initialize rate limiter
        
        Args:
            max_requests_per_day: Maximum requests per 24-hour period
            min_seconds_between_requests: Minimum seconds between requests
        """
        self.max_requests_per_day = max_requests_per_day
        self.min_seconds_between_requests = min_seconds_between_requests
        
        # Track requests per user/session
        self.request_times: Dict[str, list] = defaultdict(list)
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.last_request_time: Dict[str, float] = defaultdict(lambda: 0)
    
    def is_allowed(self, user_id: str = "default") -> Dict[str, any]:
        """
        Check if request is allowed
        
        Args:
            user_id: User or session identifier
        
        Returns:
            Dict with 'allowed' boolean and status info
        """
        current_time = time.time()
        current_datetime = datetime.now()
        
        # Clean old requests (older than 24 hours)
        cutoff_time = current_time - (24 * 3600)
        self.request_times[user_id] = [t for t in self.request_times[user_id] if t > cutoff_time]
        
        # Check daily limit
        if len(self.request_times[user_id]) >= self.max_requests_per_day:
            reset_time = self.request_times[user_id][0] + (24 * 3600)
            reset_datetime = datetime.fromtimestamp(reset_time)
            
            return {
                "allowed": False,
                "reason": "Daily limit exceeded",
                "requests_today": len(self.request_times[user_id]),
                "limit": self.max_requests_per_day,
                "reset_time": reset_datetime.isoformat(),
                "reset_in_seconds": int(reset_time - current_time)
            }
        
        # Check rate limit (minimum seconds between requests)
        time_since_last = current_time - self.last_request_time[user_id]
        if time_since_last < self.min_seconds_between_requests:
            wait_time = self.min_seconds_between_requests - time_since_last
            
            return {
                "allowed": False,
                "reason": "Rate limit exceeded",
                "wait_seconds": round(wait_time, 1),
                "min_seconds_between_requests": self.min_seconds_between_requests
            }
        
        # Request is allowed
        self.request_times[user_id].append(current_time)
        self.last_request_time[user_id] = current_time
        
        return {
            "allowed": True,
            "requests_today": len(self.request_times[user_id]),
            "remaining_today": self.max_requests_per_day - len(self.request_times[user_id]),
            "limit": self.max_requests_per_day
        }
    
    def get_status(self, user_id: str = "default") -> Dict[str, any]:
        """Get rate limit status for user"""
        current_time = time.time()
        cutoff_time = current_time - (24 * 3600)
        
        # Clean old requests
        self.request_times[user_id] = [t for t in self.request_times[user_id] if t > cutoff_time]
        
        requests_today = len(self.request_times[user_id])
        remaining = self.max_requests_per_day - requests_today
        
        # Calculate reset time
        if requests_today > 0:
            reset_time = self.request_times[user_id][0] + (24 * 3600)
            time_until_reset = reset_time - current_time
        else:
            time_until_reset = 0
        
        return {
            "requests_today": requests_today,
            "limit": self.max_requests_per_day,
            "remaining": remaining,
            "percentage_used": round((requests_today / self.max_requests_per_day) * 100, 1),
            "reset_in_hours": round(max(0, time_until_reset / 3600), 1)
        }
    
    def reset_user(self, user_id: str = "default"):
        """Reset limit for a user (for testing)"""
        self.request_times[user_id] = []
        self.last_request_time[user_id] = 0
        logger.info(f"Rate limit reset for user: {user_id}")
    
    def get_all_stats(self) -> Dict[str, any]:
        """Get stats for all users"""
        stats = {}
        for user_id in self.request_times.keys():
            stats[user_id] = self.get_status(user_id)
        return stats


class TokenBucket:
    """
    Token bucket algorithm for more flexible rate limiting
    Allows burst traffic within limits
    """
    
    def __init__(self, capacity: int = 50, refill_rate: float = 1.0):
        """
        Initialize token bucket
        
        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens
        
        Args:
            tokens: Number of tokens to consume
        
        Returns:
            True if allowed, False otherwise
        """
        # Refill tokens
        now = time.time()
        elapsed = now - self.last_refill
        
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now
        
        # Try to consume
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    def get_status(self) -> Dict[str, any]:
        """Get bucket status"""
        # Refill first
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        current_tokens = min(self.capacity, self.tokens + new_tokens)
        
        return {
            "tokens_available": round(current_tokens, 2),
            "capacity": self.capacity,
            "refill_rate": self.refill_rate,
            "percentage_full": round((current_tokens / self.capacity) * 100, 1)
        }


# Global rate limiter instances
_rate_limiter = None
_token_bucket = None


def get_rate_limiter(max_requests_per_day: int = 50, min_seconds: int = 2) -> RateLimiter:
    """Get or create rate limiter (singleton)"""
    global _rate_limiter
    
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(max_requests_per_day, min_seconds)
    
    return _rate_limiter


def get_token_bucket(capacity: int = 50, refill_rate: float = 1.0) -> TokenBucket:
    """Get or create token bucket (singleton)"""
    global _token_bucket
    
    if _token_bucket is None:
        _token_bucket = TokenBucket(capacity, refill_rate)
    
    return _token_bucket


# Convenience functions
def check_rate_limit(user_id: str = "default") -> Dict:
    """Quick rate limit check"""
    limiter = get_rate_limiter()
    return limiter.is_allowed(user_id)


def get_rate_limit_status(user_id: str = "default") -> Dict:
    """Quick status check"""
    limiter = get_rate_limiter()
    return limiter.get_status(user_id)