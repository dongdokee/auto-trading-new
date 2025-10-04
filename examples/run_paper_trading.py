#!/usr/bin/env python3
"""
Simple Paper Trading Example
간단한 모의매매 실행 예제

이 스크립트는 paper trading 시스템을 빠르게 시작할 수 있는 간단한 예제입니다.
초보자도 쉽게 사용할 수 있도록 최소한의 설정으로 구성되어 있습니다.
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Configure logging for this example
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def check_environment():
    """환경 설정 확인"""
    print("🔍 환경 설정 확인 중...")

    required_vars = [
        'BINANCE_TESTNET_API_KEY',
        'BINANCE_TESTNET_API_SECRET'
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print("❌ 다음 환경변수가 설정되지 않았습니다:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n📝 설정 방법:")
        print("1. .env 파일 생성: cp .env.template .env")
        print("2. .env 파일에서 API 키 설정")
        print("3. 또는 직접 환경변수 설정:")
        print(f"   export BINANCE_TESTNET_API_KEY='your_key'")
        print(f"   export BINANCE_TESTNET_API_SECRET='your_secret'")
        return False

    print("✅ 환경 설정 완료")
    return True


def check_config_file():
    """설정 파일 확인"""
    config_file = project_root / "config" / "trading.yaml"
    if not config_file.exists():
        print(f"❌ 설정 파일을 찾을 수 없습니다: {config_file}")
        print("📝 config/trading.yaml 파일이 필요합니다.")
        return False

    print("✅ 설정 파일 확인 완료")
    return True


async def run_simple_paper_trading():
    """간단한 paper trading 실행"""
    try:
        # Import the main paper trading system
        from scripts.paper_trading import PaperTradingSystem

        print("🚀 간단한 Paper Trading 시작...")
        print("=" * 50)

        # Create paper trading system with default config
        config_path = project_root / "config" / "trading.yaml"
        paper_trading = PaperTradingSystem(str(config_path))

        # Initialize
        await paper_trading.initialize()

        print("\n📊 Paper Trading 정보:")
        print(f"   💰 시작 자금: ${paper_trading.virtual_balance:,.2f}")
        print(f"   📈 거래 가능 종목: {', '.join(paper_trading.config.get('trading', {}).get('trading_pairs', []))}")
        print(f"   🎯 세션 ID: {paper_trading.session_id}")
        print("\n⏰ 시스템이 실행 중입니다. Ctrl+C로 종료할 수 있습니다.")
        print("📈 실시간 거래 신호를 모니터링하고 있습니다...\n")

        # Run paper trading
        await paper_trading.run()

    except KeyboardInterrupt:
        print("\n⏹️  사용자가 중단했습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        logger.error(f"Paper trading error: {e}", exc_info=True)


def create_minimal_config():
    """최소한의 설정 파일 생성 (설정 파일이 없는 경우)"""
    config_dir = project_root / "config"
    config_file = config_dir / "trading.yaml"

    if config_file.exists():
        return True

    print("📝 기본 설정 파일을 생성합니다...")

    # Create config directory if it doesn't exist
    config_dir.mkdir(exist_ok=True)

    minimal_config = """# Minimal Paper Trading Configuration
trading:
  mode: "paper"
  trading_pairs:
    - "BTC/USDT"
    - "ETH/USDT"

exchanges:
  binance:
    api_key: "${BINANCE_TESTNET_API_KEY}"
    api_secret: "${BINANCE_TESTNET_API_SECRET}"
    testnet: true
    paper_trading: true

paper_trading:
  initial_balance: 100000.0
  commission_rate: 0.001
  slippage_simulation: true
  max_slippage: 0.002
  report_interval_minutes: 15

strategies:
  momentum:
    enabled: true
    allocation: 0.5
  mean_reversion:
    enabled: true
    allocation: 0.5

risk_management:
  max_position_size: 0.1
  max_daily_loss: 0.05

logging:
  level: "INFO"
  console_handler:
    enabled: true
"""

    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(minimal_config)
        print(f"✅ 기본 설정 파일 생성 완료: {config_file}")
        return True
    except Exception as e:
        print(f"❌ 설정 파일 생성 실패: {e}")
        return False


def print_welcome_message():
    """환영 메시지 출력"""
    print("🎯 AutoTrading Paper Trading 예제")
    print("=" * 40)
    print("📝 이 예제는 Binance Testnet을 사용한 안전한 모의매매입니다.")
    print("💰 실제 돈을 사용하지 않으므로 안전하게 테스트할 수 있습니다.")
    print("🏦 Binance Testnet API 키가 필요합니다.")
    print()


def print_help():
    """도움말 출력"""
    print("📚 Paper Trading 시작 가이드:")
    print()
    print("1️⃣ Binance Testnet 계정 생성:")
    print("   https://testnet.binancefuture.com/")
    print()
    print("2️⃣ API 키 생성:")
    print("   - Testnet 로그인 후 API Management에서 생성")
    print("   - 'Enable Reading'과 'Enable Futures' 권한 활성화")
    print()
    print("3️⃣ 환경변수 설정:")
    print("   export BINANCE_TESTNET_API_KEY='your_api_key'")
    print("   export BINANCE_TESTNET_API_SECRET='your_secret_key'")
    print()
    print("4️⃣ 실행:")
    print("   python examples/run_paper_trading.py")
    print()
    print("📖 자세한 가이드: PAPER_TRADING_GUIDE.md")


async def main():
    """메인 함수"""
    print_welcome_message()

    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        print_help()
        return 0

    # Pre-flight checks
    if not check_environment():
        print("\n📖 자세한 설정 가이드는 PAPER_TRADING_GUIDE.md를 참조하세요.")
        return 1

    if not check_config_file():
        if not create_minimal_config():
            return 1

    # Load environment variables from .env file if it exists
    env_file = project_root / ".env"
    if env_file.exists():
        print("📁 .env 파일에서 환경변수 로드 중...")
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            print("✅ 환경변수 로드 완료")
        except ImportError:
            print("⚠️  python-dotenv가 설치되지 않음 (선택사항)")
        except Exception as e:
            print(f"⚠️  .env 파일 로드 중 오류: {e}")

    # Run paper trading
    try:
        await run_simple_paper_trading()
        return 0
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 안전하게 종료되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 심각한 오류: {e}")
        sys.exit(1)