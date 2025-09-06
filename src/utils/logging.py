# src/utils/logging.py

import logging
import logging.handlers
import sys
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import contextmanager


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
            
        return json.dumps(log_entry, default=str)


class StructuredLogger:
    """Enhanced logger with structured logging capabilities."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs) -> None:
        """Set persistent context for all log messages."""
        self._context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear all context."""
        self._context.clear()
    
    @contextmanager
    def context(self, **kwargs):
        """Temporary context manager for logging."""
        old_context = self._context.copy()
        self._context.update(kwargs)
        try:
            yield
        finally:
            self._context = old_context
    
    def _log_with_context(self, level: int, message: str, **kwargs) -> None:
        """Log message with context."""
        extra_fields = {**self._context, **kwargs}
        extra = {'extra_fields': extra_fields} if extra_fields else {}
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        self._log_with_context(logging.ERROR, message, **kwargs)


class LoggingManager:
    """Centralized logging configuration and management."""
    
    def __init__(self):
        self.configured = False
        self.handlers = []
    
    def setup_logging(
        self,
        level: str = "INFO",
        log_file: Optional[str] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        json_format: bool = False,
        enable_console: bool = True
    ) -> None:
        """
        Setup comprehensive logging configuration.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (optional)
            max_file_size: Maximum size per log file in bytes
            backup_count: Number of backup files to keep
            json_format: Whether to use JSON formatting
            enable_console: Whether to enable console logging
        """
        if self.configured:
            return
            
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Set logging level
        log_level = getattr(logging, level.upper(), logging.INFO)
        root_logger.setLevel(log_level)
        
        # Choose formatter
        if json_format:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
            self.handlers.append(console_handler)
        
        # File handler with rotation
        if log_file:
            try:
                # Ensure log directory exists
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=max_file_size,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                file_handler.setLevel(log_level)
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
                self.handlers.append(file_handler)
                
            except Exception as e:
                logging.error(f"Failed to setup file logging: {e}")
        
        # Mark as configured
        self.configured = True
        
        logging.info(f"Logging configured: level={level}, file={log_file}, json={json_format}")
    
    def get_structured_logger(self, name: str) -> StructuredLogger:
        """Get a structured logger instance."""
        return StructuredLogger(name)
    
    def shutdown(self) -> None:
        """Shutdown logging and cleanup handlers."""
        for handler in self.handlers:
            try:
                handler.close()
            except Exception as e:
                print(f"Error closing log handler: {e}")
        
        self.handlers.clear()
        self.configured = False


# Global logging manager instance
_logging_manager = LoggingManager()

# Convenience functions
def setup_logging(**kwargs) -> None:
    """Setup logging with the global manager."""
    _logging_manager.setup_logging(**kwargs)

def get_structured_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return _logging_manager.get_structured_logger(name)

def shutdown_logging() -> None:
    """Shutdown logging."""
    _logging_manager.shutdown()


# Performance monitoring decorator
def log_performance(logger_name: str = __name__):
    """Decorator to log function performance metrics."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_structured_logger(logger_name)
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                logger.info(
                    f"Function {func.__name__} completed",
                    execution_time_seconds=execution_time,
                    function_name=func.__name__
                )
                
                return result
                
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                
                logger.error(
                    f"Function {func.__name__} failed",
                    execution_time_seconds=execution_time,
                    function_name=func.__name__,
                    error=str(e)
                )
                raise
                
        return wrapper
    return decorator


# Async performance monitoring decorator
def log_async_performance(logger_name: str = __name__):
    """Decorator to log async function performance metrics."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            logger = get_structured_logger(logger_name)
            start_time = datetime.now()
            
            try:
                result = await func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                logger.info(
                    f"Async function {func.__name__} completed",
                    execution_time_seconds=execution_time,
                    function_name=func.__name__
                )
                
                return result
                
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                
                logger.error(
                    f"Async function {func.__name__} failed",
                    execution_time_seconds=execution_time,
                    function_name=func.__name__,
                    error=str(e)
                )
                raise
                
        return wrapper
    return decorator