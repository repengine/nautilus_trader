#!/usr/bin/env python3
"""
Simple FRED data updater - handles empty data gracefully.
"""
import os
import pandas as pd
import fredapi
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

def update_fred_data():
    """Update FRED data with latest values."""
    # Check API key
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        print("❌ FRED_API_KEY not found in environment")
        print("Run: export FRED_API_KEY='your_key' or activate your venv")
        return False
    
    fred = fredapi.Fred(api_key=api_key)
    
    # Focus on daily indicators that actually update frequently
    daily_indicators = {
        'VIXCLS': 'VIX Index',
        'DGS1': '1Y Treasury',
        'DGS2': '2Y Treasury', 
        'DGS10': '10Y Treasury',
        'DGS30': '30Y Treasury',
        'SOFR': 'SOFR Rate',
        'DTWEXBGS': 'Dollar Index',
        'DEXUSEU': 'USD/EUR',
        'BAMLH0A0HYM2': 'High Yield Spread',
        'BAMLC0A0CM': 'Investment Grade Spread',
        'MORTGAGE30US': '30Y Mortgage Rate'
    }
    
    # Get last 60 days to ensure we capture latest data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    print("📊 FRED Data Update")
    print("=" * 40)
    print(f"Fetching data from {start_date.date()} to {end_date.date()}")
    print()
    
    updated_data = {}
    success_count = 0
    
    for series_id, name in daily_indicators.items():
        try:
            # Fetch recent data
            data = fred.get_series(series_id, start_date, end_date)
            
            if data.empty or data.dropna().empty:
                print(f"⚠️  {name}: No recent data")
                continue
                
            # Get latest non-null value
            latest_data = data.dropna()
            latest_value = latest_data.iloc[-1]
            latest_date = latest_data.index[-1]
            
            updated_data[series_id] = {
                'value': latest_value,
                'date': latest_date,
                'data': data
            }
            
            print(f"✅ {name}: {latest_value:.2f} (as of {latest_date.strftime('%Y-%m-%d')})")
            success_count += 1
            
        except Exception as e:
            print(f"❌ {name}: {str(e)[:50]}...")
            continue
    
    print()
    print(f"📈 Updated {success_count}/{len(daily_indicators)} indicators")
    
    # Save to simple CSV for now (can integrate with DataStore later)
    if updated_data:
        # Create a simple combined DataFrame
        combined_data = []
        
        for series_id, info in updated_data.items():
            data = info['data'].dropna()
            for date, value in data.items():
                combined_data.append({
                    'date': date,
                    'series_id': series_id,
                    'value': value
                })
        
        if combined_data:
            df = pd.DataFrame(combined_data)
            df['timestamp_ns'] = pd.to_datetime(df['date']).astype('int64')
            
            # Pivot to wide format
            wide_df = df.pivot(index='date', columns='series_id', values='value')
            wide_df.reset_index(inplace=True)
            wide_df['timestamp_ns'] = pd.to_datetime(wide_df['date']).astype('int64')
            
            # Save updated data
            output_file = 'data/fred/fred_indicators_updated.parquet'
            os.makedirs('data/fred', exist_ok=True)
            wide_df.to_parquet(output_file)
            
            print(f"💾 Saved {len(wide_df)} rows to {output_file}")
            print()
            print("📋 Latest Values Summary:")
            for series_id, info in updated_data.items():
                name = daily_indicators[series_id]
                print(f"  {name}: {info['value']:.2f}")
    
    return success_count > 0

if __name__ == "__main__":
    success = update_fred_data()
    
    if success:
        print("\n✅ FRED data update completed successfully!")
        print("\nNext steps:")
        print("1. Use updated data in ML models")
        print("2. Set up daily cron job for automatic updates")
    else:
        print("\n❌ FRED data update failed")