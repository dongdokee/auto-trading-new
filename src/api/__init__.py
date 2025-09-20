# src/api/__init__.py
"""
API module for exchange connectivity and market data streaming.
Provides unified interface for cryptocurrency exchange integration.
"""

from .base import BaseExchangeClient

__all__ = ['BaseExchangeClient']