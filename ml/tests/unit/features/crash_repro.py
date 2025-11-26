
import numpy as np
import polars as pl
from ml.features.config import FeatureConfig
from ml.features.facade import FeatureEngineer

def test_crash_repro():
    print("Starting test_crash_repro")
    config = FeatureConfig()
    print("Config created")
    batch_engineer = FeatureEngineer(config)
    print("Engineer created")

    mock_bars_data = []
    for i in range(100):
        mock_bars_data.append({
            "close": 1.1 + i*0.001,
            "high": 1.11 + i*0.001,
            "low": 1.09 + i*0.001,
            "volume": 1000.0,
            "ts_event": i * 60_000_000_000,
        })
    bars_df = pl.DataFrame(mock_bars_data)
    print("DataFrame created")

    _batch_features, _ = batch_engineer.calculate_features_batch(bars_df)
    print("Batch features calculated")

    from ml.features.indicators import IndicatorManager
    from nautilus_trader.model.data import Bar, BarType, BarSpecification
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.enums import BarAggregation, PriceType, AggressorSide
    from nautilus_trader.model.objects import Price, Quantity

    indicator_mgr = IndicatorManager(config)
    online_engineer = FeatureEngineer(config)

    instrument_id = InstrumentId.from_str("TEST.USD")
    bar_type = BarType(instrument_id, BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST))

    for bar_data in mock_bars_data:
        current_bar = {
            "open": bar_data.get("open", bar_data["close"]),
            "high": bar_data["high"],
            "low": bar_data["low"],
            "close": bar_data["close"],
            "volume": bar_data["volume"],
        }

        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str(str(current_bar["open"])),
            high=Price.from_str(str(current_bar["high"])),
            low=Price.from_str(str(current_bar["low"])),
            close=Price.from_str(str(current_bar["close"])),
            volume=Quantity.from_str(str(current_bar["volume"])),
            ts_event=int(bar_data["ts_event"]),
            ts_init=int(bar_data["ts_event"]),
        )
        indicator_mgr.update_from_bar(bar)

        features = online_engineer.calculate_features_online(
            current_bar=current_bar,
            indicator_manager=indicator_mgr,
        )
    print("Online features calculated")

if __name__ == "__main__":
    test_crash_repro()
