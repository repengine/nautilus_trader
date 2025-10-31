# PhaseOne Signal Parity

Rows compared: 2939

## Columns only in component output

- PAYEMS_delta_1d
- UNRATE_delta_1d
- close

All shared numeric columns matched exactly.

## PhaseOne feature families detected

- Macro Delta Columns (2): PAYEMS_delta_1d, UNRATE_delta_1d
- Calendar Lag Columns (27): days_since_last_cpi, days_since_last_earnings, days_since_last_fed, days_to_next_cpi, days_to_next_earnings, days_to_next_fed, days_to_next_holiday, earnings_within_1h, earnings_within_24h, earnings_within_4h, earnings_within_72h, economic_release_within_1h, economic_release_within_24h, economic_release_within_4h, economic_release_within_72h, fed_meeting_within_1h, fed_meeting_within_24h, fed_meeting_within_4h, fed_meeting_within_72h, hours_to_earnings, hours_to_economic_release, hours_to_fed_meeting, hours_to_options_expiry, options_expiry_within_1h, options_expiry_within_24h, options_expiry_within_4h, options_expiry_within_72h
- Clustering Tag Columns (5): event_clustering_score, event_density_24h, event_density_week, total_events_24h, total_events_week
- Context Feature Columns (20): event_importance_score, has_cpi_event_today, has_earnings_in_24h, has_earnings_in_week, has_earnings_today, has_economic_release_in_24h, has_economic_release_in_week, has_fed_event_today, has_fed_meeting_in_24h, has_fed_meeting_in_week, has_options_expiry_in_24h, has_options_expiry_in_week, is_aftermarket, is_earnings_season, is_fomc_week, is_holiday_week, is_macro_available, is_market_open, is_premarket, is_triple_witching

## Capability flags

- include_calendar = False
- include_calendar_lags = True
- include_clustering_tags = True
- include_context_features = True
- include_earnings = False
- include_events = True
- include_l2 = False
- include_macro = True
- include_macro_deltas = True
- include_macro_revisions = False
- include_micro = False
- student_mode = False

## PhaseOne metadata snapshot

- macro_delta_columns (2): PAYEMS_delta_1d, UNRATE_delta_1d
- calendar_lag_columns (27): days_since_last_cpi, days_since_last_earnings, days_since_last_fed, days_to_next_cpi, days_to_next_earnings, days_to_next_fed, days_to_next_holiday, earnings_within_1h, earnings_within_24h, earnings_within_4h, earnings_within_72h, economic_release_within_1h, economic_release_within_24h, economic_release_within_4h, economic_release_within_72h, fed_meeting_within_1h, fed_meeting_within_24h, fed_meeting_within_4h, fed_meeting_within_72h, hours_to_earnings, hours_to_economic_release, hours_to_fed_meeting, hours_to_options_expiry, options_expiry_within_1h, options_expiry_within_24h, options_expiry_within_4h, options_expiry_within_72h
- clustering_tag_columns (5): event_clustering_score, event_density_24h, event_density_week, total_events_24h, total_events_week
- context_feature_columns (20): event_importance_score, has_cpi_event_today, has_earnings_in_24h, has_earnings_in_week, has_earnings_today, has_economic_release_in_24h, has_economic_release_in_week, has_fed_event_today, has_fed_meeting_in_24h, has_fed_meeting_in_week, has_options_expiry_in_24h, has_options_expiry_in_week, is_aftermarket, is_earnings_season, is_fomc_week, is_holiday_week, is_macro_available, is_market_open, is_premarket, is_triple_witching
