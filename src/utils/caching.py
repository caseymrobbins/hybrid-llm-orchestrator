# src/utils/caching.py

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union
import diskcache as dc

logger = logging.getLogger(__name__)

class CacheError(Exception):
    """Custom exception for cache-related errors."""
    pass

class Cache:
    """
    A persistent on-disk cache for LLM responses with expiration and size limits.
    
    Features:
    - SHA256 hashing for keys
    - Configurable expiration times
    - Size limits with LRU eviction
    - Async-safe operations
    - Statistics tracking
    """

    def __init__(
        self, 
        cache_dir: str = ".cache",
        max_size: int = 1024 * 1024 * 1024,  # 1GB default
        default_expire: int = 86400,  # 24 hours default
        eviction_policy: str = "least-recently-used"
    ):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_size: Maximum cache size in bytes
            default_expire: Default expiration time in seconds
            eviction_policy: Cache eviction policy
        """
        self.cache_dir = Path(cache_dir)
        self.default_expire = default_expire
        self.stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "sets": 0
        }
        
        try:
            # Create cache directory
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize diskcache with configuration
            self.cache = dc.Cache(
                directory=str(self.cache_dir),
                size_limit=max_size,
                eviction_policy=eviction_policy
            )
            
            logger.info(f"Cache initialized at {self.cache_dir} (max_size: {max_size} bytes)")
            
        except Exception as e:
            raise CacheError(f"Failed to initialize cache: {e}")

    def _generate_key(self, text: str, prefix: str = "") -> str:
        """
        Create a SHA256 hash of the input text to use as a cache key.
        
        Args:
            text: Input text to hash
            prefix: Optional prefix for the key
            
        Returns:
            SHA256 hash as hex string
        """
        try:
            key_input = f"{prefix}{text}".encode('utf-8')
            return hashlib.sha256(key_input).hexdigest()
        except Exception as e:
            logger.error(f"Failed to generate cache key: {e}")
            raise CacheError(f"Key generation failed: {e}")

    async def get(self, key_text: str, prefix: str = "") -> Optional[str]:
        """
        Retrieve an item from the cache asynchronously.
        
        Args:
            key_text: Text to use for key generation
            prefix: Optional prefix for cache key
            
        Returns:
            Cached value if found, None otherwise
        """
        try:
            key = self._generate_key(key_text, prefix)
            
            # Run cache operation in thread pool to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._sync_get, key
            )
            
            if result is not None:
                self.stats["hits"] += 1
                logger.debug(f"Cache hit for key: {key[:16]}...")
                return result
            else:
                self.stats["misses"] += 1
                logger.debug(f"Cache miss for key: {key[:16]}...")
                return None
                
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache get error: {e}")
            return None

    def _sync_get(self, key: str) -> Optional[str]:
        """Synchronous cache get operation."""
        try:
            return self.cache.get(key)
        except Exception as e:
            logger.error(f"Disk cache get error: {e}")
            return None

    async def set(
        self, 
        key_text: str, 
        value: str, 
        expire: Optional[int] = None,
        prefix: str = ""
    ) -> bool:
        """
        Save an item to the cache asynchronously.
        
        Args:
            key_text: Text to use for key generation
            value: Value to cache
            expire: Expiration time in seconds (None for default)
            prefix: Optional prefix for cache key
            
        Returns:
            True if successfully cached, False otherwise
        """
        try:
            key = self._generate_key(key_text, prefix)
            expire_time = expire or self.default_expire
            
            # Run cache operation in thread pool
            success = await asyncio.get_event_loop().run_in_executor(
                None, self._sync_set, key, value, expire_time
            )
            
            if success:
                self.stats["sets"] += 1
                logger.debug(f"Cache set for key: {key[:16]}... (expire: {expire_time}s)")
            else:
                self.stats["errors"] += 1
                
            return success
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache set error: {e}")
            return False

    def _sync_set(self, key: str, value: str, expire: int) -> bool:
        """Synchronous cache set operation."""
        try:
            return self.cache.set(key, value, expire=expire)
        except Exception as e:
            logger.error(f"Disk cache set error: {e}")
            return False

    async def delete(self, key_text: str, prefix: str = "") -> bool:
        """
        Delete an item from the cache.
        
        Args:
            key_text: Text to use for key generation
            prefix: Optional prefix for cache key
            
        Returns:
            True if item was deleted, False otherwise
        """
        try:
            key = self._generate_key(key_text, prefix)
            
            success = await asyncio.get_event_loop().run_in_executor(
                None, self._sync_delete, key
            )
            
            if success:
                logger.debug(f"Cache delete for key: {key[:16]}...")
            
            return success
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    def _sync_delete(self, key: str) -> bool:
        """Synchronous cache delete operation."""
        try:
            return self.cache.delete(key)
        except Exception as e:
            logger.error(f"Disk cache delete error: {e}")
            return False

    async def clear(self) -> bool:
        """
        Clear all items from the cache.
        
        Returns:
            True if cache was cleared successfully, False otherwise
        """
        try:
            await asyncio.get_event_loop().run_in_executor(None, self.cache.clear)
            logger.info("Cache cleared successfully")
            
            # Reset statistics
            self.stats = {
                "hits": 0,
                "misses": 0,
                "errors": 0,
                "sets": 0
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics and information.
        
        Returns:
            Dictionary containing cache statistics
        """
        try:
            # Get cache volume information
            volume_info = dict(self.cache.volume())
            
            # Calculate hit rate
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "statistics": {
                    **self.stats,
                    "hit_rate_percent": round(hit_rate, 2),
                    "total_requests": total_requests
                },
                "volume": volume_info,
                "configuration": {
                    "cache_dir": str(self.cache_dir),
                    "default_expire": self.default_expire,
                    "max_size": getattr(self.cache, '_size_limit', 'unknown')
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {
                "statistics": self.stats,
                "error": str(e)
            }

    async def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of entries cleaned up
        """
        try:
            # Note: diskcache handles expiration automatically,
            # but we can trigger a manual cleanup
            cleaned = await asyncio.get_event_loop().run_in_executor(
                None, self._sync_cleanup
            )
            
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired cache entries")
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
            return 0

    def _sync_cleanup(self) -> int:
        """Synchronous cache cleanup operation."""
        try:
            # Get current size before cleanup
            initial_count = len(self.cache)
            
            # Trigger eviction and cleanup
            self.cache.expire()
            
            # Calculate cleaned entries
            final_count = len(self.cache)
            return max(0, initial_count - final_count)
            
        except Exception as e:
            logger.error(f"Disk cache cleanup error: {e}")
            return 0

    def __len__(self) -> int:
        """Get number of items in cache."""
        try:
            return len(self.cache)
        except Exception:
            return 0

    def close(self) -> None:
        """Close the cache and clean up resources."""
        try:
            self.cache.close()
            logger.info("Cache closed successfully")
        except Exception as e:
            logger.error(f"Error closing cache: {e}")


class SemanticCache(Cache):
    """
    Extension of Cache that provides semantic similarity-based caching.
    
    This would require a vector similarity backend for production use.
    For now, it's a placeholder that uses exact text matching.
    """
    
    def __init__(self, *args, similarity_threshold: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.similarity_threshold = similarity_threshold
        logger.warning("SemanticCache is using exact text matching. For production, implement vector similarity.")

    async def get_similar(self, key_text: str, prefix: str = "") -> Optional[str]:
        """
        Get cached value for semantically similar text.
        
        Currently implements exact matching as placeholder.
        In production, this would use vector similarity search.
        """
        # For now, fall back to exact matching
        return await self.get(key_text, prefix)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics including semantic cache info."""
        stats = super().get_stats()
        stats["semantic_config"] = {
            "similarity_threshold": self.similarity_threshold,
            "implementation": "exact_match_placeholder"
        }
        return stats