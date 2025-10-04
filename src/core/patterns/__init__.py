"""
Core patterns module for common design patterns.

This module provides standardized base classes and utilities for:
- Connection management
- Logging factory
- Manager lifecycle management
"""

from .connection import BaseConnectionManager
from .logging import LoggerFactory
from .manager import BaseManager

__all__ = [
    "BaseConnectionManager",
    "LoggerFactory",
    "BaseManager"
]