# Validation Reports: Coverage Alerts Snapshot

This directory stores lightweight artefacts generated from the 3D risk pipeline validation run. The
`coverage_alerts_example.json` file mirrors the structure emitted by:

```
python -m playground.risk_model.cli \
  --start 2020-01-01 --end 2024-12-31 \
  --coverage-report playground/tests/validation_reports/coverage_alerts_example.json \
  --persist-dir playground/data/sector_dataset \
  --visualization-dir playground/data/visualizations
```

Key fields:
- `coverage.composite_coverage` now includes macro-composite ratios surfaced to the Three.js view.
- `coverage_alerts.composite` reflects the same deficits rendered as badges/tooltips in
  `playground/portfolio-3d-risk.html`.

Re-run the command after pipeline changes to refresh the snapshot prior to releasing updated
visualisation payloads.
