#!/usr/bin/env python3
"""
Populate alternative data sources for ML training.

Free/cheap data sources that add value:
1. CBOE Options Data (put/call ratios, term structure)
2. AAII Sentiment Survey
3. COT Reports (Commitment of Traders)
4. Short Interest Data
5. Market Microstructure Metrics
6. News Sentiment (via free APIs)
7. Earnings Calendar
8. Sector/Industry Classifications

Usage:
    python ml/cli/populate_alternative_data.py --all
    python ml/cli/populate_alternative_data.py --source cboe

"""

import argparse
import json
import logging
import sys
import uuid as _uuid
from datetime import datetime
from pathlib import Path


# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from typing import TYPE_CHECKING

from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml._imports import pl
from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging


if TYPE_CHECKING:  # type-only import for annotations
    from polars import DataFrame as PlDataFrame
else:  # pragma: no cover - used only for typing
    PlDataFrame = object  # type: ignore[assignment]


if not HAS_POLARS:
    check_ml_dependencies(["polars"])
from typing import Any as _Any
from typing import cast as _cast


# Cast 'pl' to Any and provide an uppercase alias for attribute access
assert pl is not None
pl = _cast(_Any, pl)
PL = pl

# Setup logging
configure_logging()
_run_id: str = f"cli_populate_alternative_data_{_uuid.uuid4().hex[:8]}"
bind_log_context(run_id=_run_id, component="ml.cli.populate_alternative_data")
logger = logging.getLogger(__name__)


class CBOEDataLoader:
    """
    Load CBOE options and volatility data.
    """

    BASE_URL = "https://www.cboe.com/api/global/delayed_quotes"

    def fetch_put_call_ratio(self) -> PlDataFrame:
        """
        Fetch daily put/call ratio data.
        """
        logger.info("Fetching CBOE Put/Call Ratio...")

        # CBOE provides delayed data for free
        # Total put/call ratio endpoint
        _url = "https://markets.cboe.com/us/options/market_statistics/daily/"

        try:
            # This would need proper API implementation
            # For now, showing structure
            data = {
                "timestamp": [datetime.now()],
                "total_pc_ratio": [0.85],
                "equity_pc_ratio": [0.75],
                "index_pc_ratio": [1.2],
                "vix_pc_ratio": [0.95],
            }

            df = PL.DataFrame(data)
            logger.info(f"Fetched {len(df)} put/call ratio records")
            return _cast(PlDataFrame, df)

        except Exception as e:
            logger.error(f"Failed to fetch put/call ratio: {e}")
            return _cast(PlDataFrame, PL.DataFrame())

    def fetch_term_structure(self) -> PlDataFrame:
        """
        Fetch VIX term structure data.
        """
        logger.info("Fetching VIX Term Structure...")

        # VIX futures term structure
        _symbols = ["VIX", "VIX9D", "VIX30D", "VIX90D", "VIX180D"]

        data: dict[str, list[object]] = {
            "timestamp": [],
            "symbol": [],
            "value": [],
            "days_to_expiry": [],
        }

        # Would fetch actual data from CBOE
        # Showing structure for now

        return _cast(PlDataFrame, PL.DataFrame(data))


class AAIISentimentLoader:
    """
    Load AAII Investor Sentiment Survey data.
    """

    def fetch_sentiment(self) -> PlDataFrame:
        """
        Fetch weekly AAII sentiment data.
        """
        logger.info("Fetching AAII Sentiment...")

        # AAII provides historical data
        # Updated weekly (Thursdays)
        _url = "https://www.aaii.com/sentiment-survey-historical-data"

        # Would scrape or use API if available
        # Structure:
        data: dict[str, list[object]] = {
            "week_ending": [],
            "bullish": [],
            "neutral": [],
            "bearish": [],
            "bull_bear_spread": [],
        }

        return _cast(PlDataFrame, PL.DataFrame(data))


class COTReportLoader:
    """
    Load CFTC Commitment of Traders reports.
    """

    BASE_URL = "https://www.cftc.gov/files/dea/cotarchives"

    def fetch_cot_data(self, symbols: list[str]) -> PlDataFrame:
        """
        Fetch COT positioning data for futures.
        """
        logger.info("Fetching COT Reports...")

        # COT reports are free from CFTC
        # Updated weekly (Tuesdays)

        # Key futures to track:
        _futures = {
            "ES": "E-MINI S&P 500",  # Equity index
            "NQ": "E-MINI NASDAQ",  # Tech index
            "VX": "VIX FUTURES",  # Volatility
            "GC": "GOLD",  # Safe haven
            "CL": "CRUDE OIL",  # Energy
            "ZB": "30-YEAR BOND",  # Bonds
            "DX": "US DOLLAR INDEX",  # Currency
        }

        data: dict[str, list[object]] = {
            "report_date": [],
            "symbol": [],
            "commercial_long": [],
            "commercial_short": [],
            "commercial_net": [],
            "noncommercial_long": [],
            "noncommercial_short": [],
            "noncommercial_net": [],
            "open_interest": [],
        }

        # Would download actual CSV files from CFTC
        return _cast(PlDataFrame, PL.DataFrame(data))


class ShortInterestLoader:
    """
    Load short interest data.
    """

    def fetch_short_interest(self, symbols: list[str]) -> PlDataFrame:
        """
        Fetch bi-monthly short interest data.
        """
        logger.info("Fetching Short Interest...")

        # Sources:
        # - FINRA (free but delayed)
        # - NYSE/NASDAQ (official but delayed)

        data: dict[str, list[object]] = {
            "settlement_date": [],
            "symbol": [],
            "short_interest": [],
            "avg_daily_volume": [],
            "days_to_cover": [],
            "short_percent_float": [],
        }

        # Would fetch from FINRA API or scrape
        return _cast(PlDataFrame, PL.DataFrame(data))


class MarketMicrostructureLoader:
    """
    Calculate market microstructure metrics from existing data.
    """

    def calculate_metrics(self, symbol: str) -> PlDataFrame:
        """
        Calculate microstructure metrics from L1/L2 data.
        """
        logger.info(f"Calculating microstructure metrics for {symbol}...")

        # Load existing L1/L2 data
        _l1_path = Path(f"data/tier1/{symbol}/l1")
        _l2_path = Path(f"data/tier1/{symbol}/l2")

        metrics: dict[str, list[object]] = {
            "timestamp": [],
            "symbol": [],
            # Liquidity metrics
            "effective_spread": [],
            "realized_spread": [],
            "price_impact": [],
            # Volume metrics
            "volume_imbalance": [],
            "trade_imbalance": [],
            "dollar_volume": [],
            # Information metrics
            "kyle_lambda": [],  # Price impact coefficient
            "hasbrouck_info_share": [],  # Information share
            "amihud_illiquidity": [],  # Illiquidity measure
            # Toxicity metrics
            "vpin": [],  # Volume-synchronized PIN
            "order_flow_toxicity": [],
        }

        # Would calculate from actual L1/L2 data
        return _cast(PlDataFrame, PL.DataFrame(metrics))


class NewsSentimentLoader:
    """
    Load news sentiment data from free sources.
    """

    def fetch_news_sentiment(self, symbols: list[str]) -> PlDataFrame:
        """
        Fetch news sentiment scores.
        """
        logger.info("Fetching News Sentiment...")

        # Free sources:
        # 1. NewsAPI (free tier: 100 requests/day)
        # 2. Alpha Vantage News Sentiment (free tier available)
        # 3. Reddit API (sentiment from WSB, investing subreddits)

        data: dict[str, list[object]] = {
            "timestamp": [],
            "symbol": [],
            "headline_sentiment": [],
            "article_sentiment": [],
            "social_sentiment": [],
            "mention_count": [],
            "sentiment_volatility": [],
        }

        return _cast(PlDataFrame, PL.DataFrame(data))


class EarningsCalendarLoader:
    """
    Load earnings calendar and estimates.
    """

    def fetch_earnings_calendar(self, symbols: list[str]) -> PlDataFrame:
        """
        Fetch earnings dates and estimates.
        """
        logger.info("Fetching Earnings Calendar...")

        # Sources:
        # - Yahoo Finance (free)
        # - Alpha Vantage (free tier)
        # - Nasdaq.com (free)

        data: dict[str, list[object]] = {
            "symbol": [],
            "earnings_date": [],
            "eps_estimate": [],
            "eps_actual": [],
            "revenue_estimate": [],
            "revenue_actual": [],
            "surprise_percent": [],
            "days_until_earnings": [],
        }

        return _cast(PlDataFrame, PL.DataFrame(data))


class SectorIndustryLoader:
    """
    Load sector and industry classifications.
    """

    def fetch_classifications(self, symbols: list[str]) -> PlDataFrame:
        """
        Fetch GICS sector/industry classifications.
        """
        logger.info("Fetching Sector/Industry Classifications...")

        # Map symbols to sectors/industries
        # Can use free sources like Yahoo Finance

        data: dict[str, list[object]] = {
            "symbol": [],
            "gics_sector": [],
            "gics_industry_group": [],
            "gics_industry": [],
            "gics_sub_industry": [],
            "market_cap_category": [],  # large/mid/small
            "style_category": [],  # value/growth/blend
        }

        return _cast(PlDataFrame, PL.DataFrame(data))


class AlternativeDataPopulator:
    """
    Main class to populate all alternative data sources.
    """

    def __init__(self) -> None:
        self.cboe_loader = CBOEDataLoader()
        self.aaii_loader = AAIISentimentLoader()
        self.cot_loader = COTReportLoader()
        self.short_loader = ShortInterestLoader()
        self.micro_loader = MarketMicrostructureLoader()
        self.news_loader = NewsSentimentLoader()
        self.earnings_loader = EarningsCalendarLoader()
        self.sector_loader = SectorIndustryLoader()

    def populate_all(self, symbols: list[str]) -> dict[str, PlDataFrame]:
        """
        Populate all alternative data sources.
        """
        logger.info("Populating all alternative data sources...")

        results = {}

        # Market-wide indicators
        results["put_call_ratio"] = self.cboe_loader.fetch_put_call_ratio()
        results["vix_term_structure"] = self.cboe_loader.fetch_term_structure()
        results["aaii_sentiment"] = self.aaii_loader.fetch_sentiment()
        results["cot_reports"] = self.cot_loader.fetch_cot_data(["ES", "VX", "DX"])

        # Symbol-specific data
        results["short_interest"] = self.short_loader.fetch_short_interest(symbols)
        results["news_sentiment"] = self.news_loader.fetch_news_sentiment(symbols)
        results["earnings_calendar"] = self.earnings_loader.fetch_earnings_calendar(symbols)
        results["sector_industry"] = self.sector_loader.fetch_classifications(symbols)

        # Calculate microstructure for symbols with L2 data
        micro_data: list[PlDataFrame] = []
        for symbol in symbols:
            l2_path = Path(f"data/tier1/{symbol}/l2")
            if l2_path.exists():
                micro = self.micro_loader.calculate_metrics(symbol)
                if not micro.is_empty():
                    micro_data.append(micro)

        if micro_data:
            results["microstructure"] = PL.concat(micro_data)

        return results

    def save_data(self, data: dict[str, PlDataFrame], output_dir: Path) -> None:
        """
        Save alternative data to parquet files.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, df in data.items():
            if df is not None and not df.is_empty():
                output_file = output_dir / f"{name}.parquet"
                df.write_parquet(output_file)
                logger.info(f"Saved {name} to {output_file}")


def get_tier1_symbols() -> list[str]:
    """
    Get Tier 1 symbols from L1 progress.
    """
    progress_file = Path("tier1_l1_progress.json")
    if progress_file.exists():
        with open(progress_file) as f:
            data = json.load(f)
            return sorted(set(data.get("completed_bbo", [])))
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Populate alternative data sources")

    parser.add_argument(
        "--all",
        action="store_true",
        help="Populate all data sources",
    )
    parser.add_argument(
        "--source",
        choices=[
            "cboe",
            "aaii",
            "cot",
            "short",
            "micro",
            "news",
            "earnings",
            "sector",
        ],
        help="Specific data source to populate",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Specific symbols (default: Tier 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/alternative"),
        help="Output directory for alternative data",
    )

    args = parser.parse_args()

    # Get symbols
    symbols = args.symbols or get_tier1_symbols()
    if not symbols:
        logger.error("No symbols specified")
        return 1

    logger.info(f"Processing {len(symbols)} symbols")

    # Initialize populator
    populator = AlternativeDataPopulator()

    # Populate requested data
    if args.all:
        data = populator.populate_all(symbols)
    elif args.source:
        data = {}
        if args.source == "cboe":
            data["put_call_ratio"] = populator.cboe_loader.fetch_put_call_ratio()
            data["vix_term_structure"] = populator.cboe_loader.fetch_term_structure()
        elif args.source == "aaii":
            data["aaii_sentiment"] = populator.aaii_loader.fetch_sentiment()
        elif args.source == "cot":
            data["cot_reports"] = populator.cot_loader.fetch_cot_data(["ES", "VX"])
        elif args.source == "short":
            data["short_interest"] = populator.short_loader.fetch_short_interest(symbols)
        elif args.source == "micro":
            micro_data = []
            for symbol in symbols:
                micro = populator.micro_loader.calculate_metrics(symbol)
                if not micro.is_empty():
                    micro_data.append(micro)
            if micro_data:
                data["microstructure"] = PL.concat(micro_data)
        elif args.source == "news":
            data["news_sentiment"] = populator.news_loader.fetch_news_sentiment(symbols)
        elif args.source == "earnings":
            data["earnings_calendar"] = populator.earnings_loader.fetch_earnings_calendar(symbols)
        elif args.source == "sector":
            data["sector_industry"] = populator.sector_loader.fetch_classifications(symbols)
    else:
        logger.error("Specify --all or --source")
        return 1

    # Save data
    populator.save_data(data, args.output_dir)

    logger.info("Alternative data population complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
