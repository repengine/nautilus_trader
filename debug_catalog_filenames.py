from datetime import datetime
from pathlib import Path
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.model.data import Bar, BarType, BarSpecification
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.enums import BarAggregation, PriceType
from nautilus_trader.model.objects import Price, Quantity

def test_catalog_filenames():
    catalog_path = Path("tmp_catalog_test")
    if catalog_path.exists():
        import shutil
        shutil.rmtree(catalog_path)
    
    catalog = ParquetDataCatalog(str(catalog_path))
    
    instrument_id = InstrumentId.from_str("TEST.USD")
    bar_type = BarType(
        instrument_id=instrument_id,
        bar_spec=BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST),
        aggregation_source=2, # EXTERNAL
    )
    
    bars = [
        Bar(
            bar_type=bar_type,
            open=Price.from_str("1.0"),
            high=Price.from_str("1.0"),
            low=Price.from_str("1.0"),
            close=Price.from_str("1.0"),
            volume=Quantity.from_int(100),
            ts_event=1000000000,
            ts_init=1000000000,
        )
    ]
    
    catalog.write_data(bars)
    
    print("\nFiles in catalog:")
    for p in catalog_path.rglob("*"):
        if p.is_file():
            print(p)

if __name__ == "__main__":
    test_catalog_filenames()
