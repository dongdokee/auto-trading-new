"""
Trade Journal System

Comprehensive trade journaling and analytics system for paper and live trading.
Provides persistent storage, reporting, and analysis of trading activities.

Key Features:
- SQLite/PostgreSQL storage
- Comprehensive trade metadata
- HTML/CSV/JSON export
- Performance analytics
- Search and filtering
- Trade comparison tools
"""

import asyncio
import sqlite3
import json
import csv
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from .time_utils import get_utc_now
from .financial_math import (
    calculate_returns, calculate_sharpe_ratio, calculate_max_drawdown,
    calculate_volatility
)
import pandas as pd


@dataclass
class JournalEntry:
    """Single trade journal entry"""
    id: Optional[int] = None
    session_id: str = ""
    correlation_id: str = ""
    trade_id: str = ""

    # Trade details
    symbol: str = ""
    side: str = ""  # LONG/SHORT
    quantity: Optional[Decimal] = None
    entry_price: Optional[Decimal] = None
    exit_price: Optional[Decimal] = None

    # Timing
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Performance
    pnl: Optional[Decimal] = None
    pnl_pct: Optional[float] = None
    fees: Optional[Decimal] = None
    slippage_bps: Optional[Decimal] = None

    # Strategy context
    strategy: str = ""
    signal_strength: Optional[float] = None
    signal_type: str = ""
    market_regime: str = ""

    # Risk metrics
    risk_score: Optional[float] = None
    position_size_pct: Optional[float] = None
    leverage: Optional[Decimal] = None

    # Execution details
    execution_latency_ms: Optional[float] = None
    order_type: str = ""

    # Market context
    entry_bid: Optional[Decimal] = None
    entry_ask: Optional[Decimal] = None
    entry_spread_bps: Optional[float] = None
    volume_at_entry: Optional[Decimal] = None

    # Mode and environment
    trading_mode: str = ""  # paper/live/demo
    testnet: bool = True

    # Additional metadata
    notes: str = ""
    tags: str = ""  # JSON string of tags
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Post-initialization processing"""
        if self.created_at is None:
            self.created_at = get_utc_now()
        if self.updated_at is None:
            self.updated_at = self.created_at

        # Calculate derived fields
        if (self.entry_time and self.exit_time and
            self.duration_seconds is None):
            self.duration_seconds = (self.exit_time - self.entry_time).total_seconds()

        if (self.pnl and self.quantity and self.entry_price and
            self.pnl_pct is None):
            notional = self.quantity * self.entry_price
            if notional > 0:
                self.pnl_pct = float(self.pnl / notional * 100)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/export"""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Decimal):
                result[key] = float(value)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JournalEntry':
        """Create from dictionary"""
        # Convert datetime strings back to datetime objects
        for field in ['entry_time', 'exit_time', 'created_at', 'updated_at']:
            if field in data and isinstance(data[field], str):
                try:
                    data[field] = datetime.fromisoformat(data[field])
                except (ValueError, TypeError):
                    data[field] = None

        # Convert numeric strings to Decimal
        for field in ['quantity', 'entry_price', 'exit_price', 'pnl', 'fees',
                     'slippage_bps', 'leverage', 'entry_bid', 'entry_ask',
                     'volume_at_entry']:
            if field in data and data[field] is not None:
                try:
                    data[field] = Decimal(str(data[field]))
                except (ValueError, TypeError):
                    data[field] = None

        return cls(**data)


class TradeJournal:
    """
    Comprehensive trade journal system

    Features:
    - Persistent storage (SQLite/PostgreSQL)
    - Trade entry management
    - Performance analytics
    - Export capabilities
    - Search and filtering
    """

    def __init__(
        self,
        database_path: str = "data/trade_journal.db",
        mode: str = "paper",
        auto_create_tables: bool = True
    ):
        """
        Initialize trade journal

        Args:
            database_path: Path to SQLite database file
            mode: Trading mode (paper/live/demo)
            auto_create_tables: Create tables if they don't exist
        """
        self.database_path = database_path
        self.mode = mode
        self.logger = logging.getLogger(f"trade_journal_{mode}")

        # Ensure directory exists
        os.makedirs(os.path.dirname(database_path), exist_ok=True)

        # Initialize database
        self.connection = None
        self._init_database(auto_create_tables)

        self.logger.info(f"TradeJournal initialized: {database_path} (mode: {mode})")

    def _init_database(self, auto_create: bool):
        """Initialize database connection and tables"""
        try:
            self.connection = sqlite3.connect(
                self.database_path,
                check_same_thread=False,
                timeout=30.0
            )
            self.connection.row_factory = sqlite3.Row

            if auto_create:
                self._create_tables()

        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise

    def _create_tables(self):
        """Create journal tables"""
        create_journal_table = """
        CREATE TABLE IF NOT EXISTS journal_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            correlation_id TEXT NOT NULL,
            trade_id TEXT NOT NULL,

            -- Trade details
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL,
            entry_price REAL,
            exit_price REAL,

            -- Timing
            entry_time TEXT,
            exit_time TEXT,
            duration_seconds REAL,

            -- Performance
            pnl REAL,
            pnl_pct REAL,
            fees REAL,
            slippage_bps REAL,

            -- Strategy context
            strategy TEXT,
            signal_strength REAL,
            signal_type TEXT,
            market_regime TEXT,

            -- Risk metrics
            risk_score REAL,
            position_size_pct REAL,
            leverage REAL,

            -- Execution details
            execution_latency_ms REAL,
            order_type TEXT,

            -- Market context
            entry_bid REAL,
            entry_ask REAL,
            entry_spread_bps REAL,
            volume_at_entry REAL,

            -- Mode and environment
            trading_mode TEXT NOT NULL,
            testnet BOOLEAN DEFAULT 1,

            -- Additional metadata
            notes TEXT,
            tags TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """

        create_session_table = """
        CREATE TABLE IF NOT EXISTS trading_sessions (
            session_id TEXT PRIMARY KEY,
            start_time TEXT NOT NULL,
            end_time TEXT,
            mode TEXT NOT NULL,
            total_trades INTEGER DEFAULT 0,
            total_pnl REAL DEFAULT 0,
            total_fees REAL DEFAULT 0,
            win_rate REAL DEFAULT 0,
            metadata TEXT,
            created_at TEXT NOT NULL
        )
        """

        # Create indexes for performance
        create_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_journal_session_id ON journal_entries(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_journal_correlation_id ON journal_entries(correlation_id)",
            "CREATE INDEX IF NOT EXISTS idx_journal_trade_id ON journal_entries(trade_id)",
            "CREATE INDEX IF NOT EXISTS idx_journal_symbol ON journal_entries(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_journal_entry_time ON journal_entries(entry_time)",
            "CREATE INDEX IF NOT EXISTS idx_journal_strategy ON journal_entries(strategy)",
            "CREATE INDEX IF NOT EXISTS idx_journal_mode ON journal_entries(trading_mode)"
        ]

        try:
            cursor = self.connection.cursor()
            cursor.execute(create_journal_table)
            cursor.execute(create_session_table)

            for index_sql in create_indexes:
                cursor.execute(index_sql)

            self.connection.commit()
            self.logger.info("Database tables created successfully")

        except Exception as e:
            self.logger.error(f"Failed to create tables: {e}")
            raise

    def add_entry(self, entry: JournalEntry) -> int:
        """Add a new journal entry"""
        if not self.connection:
            raise RuntimeError("Database not initialized")

        try:
            entry.updated_at = get_utc_now()
            entry_dict = entry.to_dict()

            # Remove 'id' if None for INSERT
            if entry_dict.get('id') is None:
                entry_dict.pop('id', None)

            # Convert boolean to integer for SQLite
            entry_dict['testnet'] = 1 if entry_dict.get('testnet') else 0

            columns = list(entry_dict.keys())
            placeholders = ['?' for _ in columns]
            values = list(entry_dict.values())

            sql = f"""
            INSERT INTO journal_entries ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            """

            cursor = self.connection.cursor()
            cursor.execute(sql, values)
            self.connection.commit()

            entry_id = cursor.lastrowid
            self.logger.debug(f"Added journal entry: {entry_id}")
            return entry_id

        except Exception as e:
            self.logger.error(f"Failed to add journal entry: {e}")
            raise

    def update_entry(self, entry_id: int, updates: Dict[str, Any]) -> bool:
        """Update an existing journal entry"""
        if not self.connection:
            raise RuntimeError("Database not initialized")

        try:
            updates['updated_at'] = get_utc_now().isoformat()

            # Convert Decimal values to float for SQLite
            for key, value in updates.items():
                if isinstance(value, Decimal):
                    updates[key] = float(value)
                elif isinstance(value, datetime):
                    updates[key] = value.isoformat()

            set_clauses = [f"{key} = ?" for key in updates.keys()]
            values = list(updates.values())
            values.append(entry_id)

            sql = f"""
            UPDATE journal_entries
            SET {', '.join(set_clauses)}
            WHERE id = ?
            """

            cursor = self.connection.cursor()
            cursor.execute(sql, values)
            self.connection.commit()

            success = cursor.rowcount > 0
            if success:
                self.logger.debug(f"Updated journal entry: {entry_id}")
            else:
                self.logger.warning(f"No entry found to update: {entry_id}")

            return success

        except Exception as e:
            self.logger.error(f"Failed to update journal entry: {e}")
            raise

    def get_entry(self, entry_id: int) -> Optional[JournalEntry]:
        """Get a specific journal entry"""
        if not self.connection:
            return None

        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM journal_entries WHERE id = ?", (entry_id,))
            row = cursor.fetchone()

            if row:
                entry_dict = dict(row)
                entry_dict['testnet'] = bool(entry_dict['testnet'])
                return JournalEntry.from_dict(entry_dict)

            return None

        except Exception as e:
            self.logger.error(f"Failed to get journal entry {entry_id}: {e}")
            return None

    def get_entries(
        self,
        session_id: Optional[str] = None,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[JournalEntry]:
        """Get journal entries with filtering"""
        if not self.connection:
            return []

        try:
            conditions = []
            params = []

            if session_id:
                conditions.append("session_id = ?")
                params.append(session_id)

            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)

            if strategy:
                conditions.append("strategy = ?")
                params.append(strategy)

            if start_date:
                conditions.append("entry_time >= ?")
                params.append(start_date.isoformat())

            if end_date:
                conditions.append("entry_time <= ?")
                params.append(end_date.isoformat())

            where_clause = ""
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)

            limit_clause = ""
            if limit:
                limit_clause = f"LIMIT {limit} OFFSET {offset}"

            sql = f"""
            SELECT * FROM journal_entries
            {where_clause}
            ORDER BY entry_time DESC
            {limit_clause}
            """

            cursor = self.connection.cursor()
            cursor.execute(sql, params)

            entries = []
            for row in cursor.fetchall():
                entry_dict = dict(row)
                entry_dict['testnet'] = bool(entry_dict['testnet'])
                entries.append(JournalEntry.from_dict(entry_dict))

            return entries

        except Exception as e:
            self.logger.error(f"Failed to get journal entries: {e}")
            return []

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary statistics for a trading session"""
        entries = self.get_entries(session_id=session_id)

        if not entries:
            return {
                'session_id': session_id,
                'total_trades': 0,
                'total_pnl': 0,
                'total_fees': 0,
                'win_rate': 0,
                'avg_trade_duration': 0,
                'symbols_traded': [],
                'strategies_used': []
            }

        # Calculate metrics
        total_trades = len(entries)
        completed_trades = [e for e in entries if e.pnl is not None]

        total_pnl = sum(e.pnl for e in completed_trades if e.pnl)
        total_fees = sum(e.fees for e in completed_trades if e.fees)

        winning_trades = len([e for e in completed_trades if e.pnl and e.pnl > 0])
        win_rate = winning_trades / len(completed_trades) if completed_trades else 0

        # Average trade duration
        durations = [e.duration_seconds for e in entries if e.duration_seconds]
        avg_duration = sum(durations) / len(durations) if durations else 0

        # Unique symbols and strategies
        symbols = list(set(e.symbol for e in entries if e.symbol))
        strategies = list(set(e.strategy for e in entries if e.strategy))

        return {
            'session_id': session_id,
            'total_trades': total_trades,
            'completed_trades': len(completed_trades),
            'total_pnl': float(total_pnl),
            'total_fees': float(total_fees),
            'net_pnl': float(total_pnl - total_fees),
            'win_rate': win_rate,
            'winning_trades': winning_trades,
            'losing_trades': len(completed_trades) - winning_trades,
            'avg_trade_duration_seconds': avg_duration,
            'symbols_traded': symbols,
            'strategies_used': strategies
        }

    def export_to_csv(
        self,
        filename: str,
        session_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> bool:
        """Export journal entries to CSV"""
        try:
            entries = self.get_entries(
                session_id=session_id,
                start_date=start_date,
                end_date=end_date
            )

            if not entries:
                self.logger.warning("No entries to export")
                return False

            # Convert to list of dictionaries
            data = [entry.to_dict() for entry in entries]

            # Write CSV
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                if data:
                    fieldnames = data[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(data)

            self.logger.info(f"Exported {len(entries)} entries to {filename}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export to CSV: {e}")
            return False

    def export_to_json(
        self,
        filename: str,
        session_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> bool:
        """Export journal entries to JSON"""
        try:
            entries = self.get_entries(
                session_id=session_id,
                start_date=start_date,
                end_date=end_date
            )

            # Convert to serializable format
            data = {
                'export_timestamp': get_utc_now().isoformat(),
                'mode': self.mode,
                'total_entries': len(entries),
                'entries': [entry.to_dict() for entry in entries]
            }

            # Add session summary if filtering by session
            if session_id:
                data['session_summary'] = self.get_session_summary(session_id)

            with open(filename, 'w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=2, ensure_ascii=False)

            self.logger.info(f"Exported {len(entries)} entries to {filename}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export to JSON: {e}")
            return False

    def generate_html_report(
        self,
        filename: str,
        session_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> bool:
        """Generate HTML trading report"""
        try:
            entries = self.get_entries(
                session_id=session_id,
                start_date=start_date,
                end_date=end_date
            )

            if not entries:
                self.logger.warning("No entries for report")
                return False

            # Calculate performance metrics
            completed_trades = [e for e in entries if e.pnl is not None]
            pnl_series = [float(e.pnl) for e in completed_trades if e.pnl]

            if len(pnl_series) > 1:
                returns_series = pd.Series(pnl_series)
                sharpe_ratio = calculate_sharpe_ratio(returns_series, risk_free_rate=0.02)
                max_dd, _ = calculate_max_drawdown(returns_series.cumsum())
                volatility = calculate_volatility(returns_series)
            else:
                sharpe_ratio = 0.0
                max_dd = 0.0
                volatility = 0.0

            # Generate HTML
            html_content = self._generate_html_template(
                entries, completed_trades, sharpe_ratio, max_dd, volatility
            )

            with open(filename, 'w', encoding='utf-8') as htmlfile:
                htmlfile.write(html_content)

            self.logger.info(f"Generated HTML report: {filename}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to generate HTML report: {e}")
            return False

    def _generate_html_template(
        self,
        entries: List[JournalEntry],
        completed_trades: List[JournalEntry],
        sharpe_ratio: float,
        max_drawdown: float,
        volatility: float
    ) -> str:
        """Generate HTML report template"""

        # Calculate summary metrics
        total_pnl = sum(e.pnl for e in completed_trades if e.pnl)
        total_fees = sum(e.fees for e in completed_trades if e.fees)
        winning_trades = len([e for e in completed_trades if e.pnl and e.pnl > 0])
        win_rate = winning_trades / len(completed_trades) if completed_trades else 0

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Trading Journal Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; margin-bottom: 20px; }}
        .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px; }}
        .metric {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; min-width: 200px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ font-size: 14px; color: #7f8c8d; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Trading Journal Report</h1>
        <p>Generated: {get_utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        <p>Mode: {self.mode.upper()}</p>
    </div>

    <div class="metrics">
        <div class="metric">
            <div class="metric-value">{len(entries)}</div>
            <div class="metric-label">Total Trades</div>
        </div>
        <div class="metric">
            <div class="metric-value {('positive' if total_pnl >= 0 else 'negative')}">${total_pnl:.2f}</div>
            <div class="metric-label">Total P&L</div>
        </div>
        <div class="metric">
            <div class="metric-value">{win_rate:.1%}</div>
            <div class="metric-label">Win Rate</div>
        </div>
        <div class="metric">
            <div class="metric-value">{sharpe_ratio:.2f}</div>
            <div class="metric-label">Sharpe Ratio</div>
        </div>
        <div class="metric">
            <div class="metric-value">{max_drawdown:.2%}</div>
            <div class="metric-label">Max Drawdown</div>
        </div>
        <div class="metric">
            <div class="metric-value">${total_fees:.2f}</div>
            <div class="metric-label">Total Fees</div>
        </div>
    </div>

    <h2>Trade Details</h2>
    <table>
        <thead>
            <tr>
                <th>Date</th>
                <th>Symbol</th>
                <th>Side</th>
                <th>Quantity</th>
                <th>Entry Price</th>
                <th>Exit Price</th>
                <th>P&L</th>
                <th>Strategy</th>
            </tr>
        </thead>
        <tbody>
"""

        for entry in entries[:100]:  # Limit to 100 most recent trades
            pnl_class = 'positive' if entry.pnl and entry.pnl > 0 else 'negative' if entry.pnl and entry.pnl < 0 else ''
            entry_time_str = entry.entry_time.strftime('%Y-%m-%d %H:%M') if entry.entry_time else 'N/A'

            html += f"""
            <tr>
                <td>{entry_time_str}</td>
                <td>{entry.symbol}</td>
                <td>{entry.side}</td>
                <td>{entry.quantity or 'N/A'}</td>
                <td>${entry.entry_price or 0:.4f}</td>
                <td>${entry.exit_price or 0:.4f}</td>
                <td class="{pnl_class}">${entry.pnl or 0:.2f}</td>
                <td>{entry.strategy}</td>
            </tr>
"""

        html += """
        </tbody>
    </table>

</body>
</html>
"""
        return html

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.logger.info("TradeJournal connection closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Convenience functions

def create_journal_entry_from_trade_metrics(trade_metrics, **additional_data) -> JournalEntry:
    """Create journal entry from TradeMetrics object"""
    from .trading_logger import TradeMetrics

    if not isinstance(trade_metrics, TradeMetrics):
        raise ValueError("Expected TradeMetrics object")

    entry_data = {
        'correlation_id': trade_metrics.correlation_id,
        'trade_id': trade_metrics.trade_id or '',
        'symbol': trade_metrics.symbol,
        'side': trade_metrics.side,
        'quantity': trade_metrics.quantity,
        'entry_price': trade_metrics.entry_price,
        'exit_price': trade_metrics.exit_price,
        'entry_time': trade_metrics.entry_time,
        'exit_time': trade_metrics.exit_time,
        'pnl': trade_metrics.pnl,
        'fees': trade_metrics.fees,
        'slippage_bps': trade_metrics.slippage_bps,
        'execution_latency_ms': trade_metrics.execution_latency_ms,
        'signal_strength': trade_metrics.signal_strength,
        'risk_score': trade_metrics.risk_score,
        **additional_data
    }

    return JournalEntry(**entry_data)


def get_paper_trading_journal(database_path: str = "data/paper_trading_journal.db") -> TradeJournal:
    """Get paper trading journal with default settings"""
    return TradeJournal(database_path=database_path, mode="paper")


def get_live_trading_journal(database_path: str = "data/live_trading_journal.db") -> TradeJournal:
    """Get live trading journal with default settings"""
    return TradeJournal(database_path=database_path, mode="live")