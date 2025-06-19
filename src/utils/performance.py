"""
Performance optimization utilities for the Auto-Wall application.
"""
import time
import threading
from functools import wraps
from PyQt6.QtCore import QTimer


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if elapsed > 0.1:  # Only log operations that take more than 100ms
            print(f"[PERF] {self.name}: {elapsed:.3f}s")


class DebouncedFunction:
    """Debounces function calls to prevent rapid successive executions."""
    
    def __init__(self, func, delay_ms=300):
        self.func = func
        self.delay_ms = delay_ms
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._execute)
        self.pending_args = None
        self.pending_kwargs = None
        self._lock = threading.Lock()
        
    def __call__(self, *args, **kwargs):
        with self._lock:
            self.pending_args = args
            self.pending_kwargs = kwargs
            
            # Restart the timer
            if self.timer.isActive():
                self.timer.stop()
            self.timer.start(self.delay_ms)
    
    def _execute(self):
        with self._lock:
            if self.pending_args is not None:
                try:
                    self.func(*self.pending_args, **self.pending_kwargs)
                finally:
                    self.pending_args = None
                    self.pending_kwargs = None


def debounce(delay_ms=300):
    """Decorator to debounce function calls."""
    def decorator(func):
        debounced = DebouncedFunction(func, delay_ms)
        return debounced
    return decorator


class ImageCache:
    """Simple LRU cache for processed images."""
    
    def __init__(self, max_size=5):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        
    def get(self, key):
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
        
    def put(self, key, value):
        if key in self.cache:
            # Update existing
            self.cache[key] = value
            self.access_order.remove(key)
            self.access_order.append(key)
        else:
            # Add new
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                oldest = self.access_order.pop(0)
                del self.cache[oldest]
            
            self.cache[key] = value
            self.access_order.append(key)
            
    def clear(self):
        self.cache.clear()
        self.access_order.clear()


def fast_hash(data):
    """Fast hash function for cache keys."""
    if isinstance(data, (list, tuple)):
        return hash(tuple(str(x) for x in data))
    return hash(str(data))
