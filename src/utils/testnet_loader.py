"""
Testnet Parameter Loader
Loads actual parameters from Binance Testnet for paper trading
"""

import os
import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional
from pathlib import Path

from src.api.binance.client import BinanceClient
from src.api.base import ExchangeConfig


class TestnetParameterLoader:
    """Load actual parameters from Binance Testnet"""

    @staticmethod
    async def load_testnet_parameters(
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load actual parameters from Binance Testnet

        Args:
            api_key: Binance Testnet API key (if not provided, reads from env)
            api_secret: Binance Testnet API secret (if not provided, reads from env)

        Returns:
            Dictionary with testnet parameters:
            - initial_balance: USDT balance from testnet
            - commission_rate: Actual commission rate (0.04% for Binance Futures)
        """
        # Get credentials from environment if not provided
        if not api_key:
            api_key = os.getenv("BINANCE_TESTNET_API_KEY")
        if not api_secret:
            api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")

        if not api_key or not api_secret:
            raise ValueError(
                "Binance Testnet API credentials not found. "
                "Please set BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET "
                "in your .env file or pass them as arguments."
            )

        # Create testnet client
        config = ExchangeConfig(
            api_key=api_key,
            api_secret=api_secret,
            testnet=True,
            timeout=30,
            rate_limit_per_minute=1200
        )

        client = BinanceClient(config)

        try:
            # Connect to testnet
            await client.connect()

            # Get account balance
            balances = await client.get_account_balance()
            usdt_balance = balances.get("USDT", Decimal("0.0"))

            # Binance Futures commission rates:
            # - Maker: 0.02% (0.0002)
            # - Taker: 0.04% (0.0004)
            # Use taker rate as default (conservative estimate)
            commission_rate = Decimal("0.0004")

            # Get actual commission from account info if available
            try:
                account_info = await client._make_authenticated_request("GET", "/fapi/v2/account")
                if "feeTier" in account_info:
                    # Binance fee tiers: 0.04% for regular users
                    fee_tier = account_info.get("feeTier", 0)
                    # Fee tier 0 = 0.04% taker, Fee tier 1 = 0.035%, etc.
                    if fee_tier == 0:
                        commission_rate = Decimal("0.0004")
                    else:
                        # Adjust based on fee tier (simplified)
                        commission_rate = Decimal("0.0004") - (Decimal(str(fee_tier)) * Decimal("0.00005"))
            except:
                # If we can't get fee tier, use default
                pass

            return {
                "initial_balance": float(usdt_balance),
                "commission_rate": float(commission_rate)
            }

        finally:
            # Always disconnect
            await client.disconnect()


async def main():
    """Test the testnet parameter loader"""
    print("Loading parameters from Binance Testnet...")
    print("-" * 50)

    # Load environment variables from .env file
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

    try:
        params = await TestnetParameterLoader.load_testnet_parameters()

        print("✓ Successfully loaded testnet parameters:")
        print(f"  Initial Balance: ${params['initial_balance']:,.2f} USDT")
        print(f"  Commission Rate: {params['commission_rate']*100:.3f}%")
        print("\nThese values will be automatically used in paper trading.")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
