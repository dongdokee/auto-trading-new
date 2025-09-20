# src/api/binance/exceptions.py
"""
Binance-specific exceptions for API error handling.
"""


class BinanceAPIError(Exception):
    """Exception raised for Binance API errors"""

    def __init__(self, message: str, code: int = None):
        super().__init__(message)
        self.message = message
        self.code = code

    def __str__(self):
        if self.code:
            return f"Binance API Error [{self.code}]: {self.message}"
        return f"Binance API Error: {self.message}"


class BinanceConnectionError(BinanceAPIError):
    """Exception raised for connection-related errors"""
    pass


class BinanceRateLimitError(BinanceAPIError):
    """Exception raised when rate limits are exceeded"""
    pass


class BinanceOrderError(BinanceAPIError):
    """Exception raised for order-related errors"""
    pass