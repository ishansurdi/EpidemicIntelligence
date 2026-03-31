# API Reference (Phase 1 Contracts)

Base path: `/api/v1`

## Health

### GET `/health`
Returns service health and model-loading status.

## Data Endpoints

### GET `/data/timeseries`
Query params:
- `country` (optional)
- `start_date` (optional, YYYY-MM-DD)
- `end_date` (optional, YYYY-MM-DD)

Returns cleaned time series rows.

### GET `/data/features`
Query params:
- `country` (optional)
- `date` (optional)

Returns engineered features used by models.

### GET `/data/graph`
Query params:
- `snapshot_date` (optional)

Returns graph nodes and weighted edges for the requested date.

## Prediction Endpoints

### POST `/predict/forecast`
Request:
```json
{
  "region_ids": ["IND", "BRA", "USA"],
  "horizon": 28,
  "confidence": true
}
```

Response:
```json
{
  "forecasts": [
    {
      "region": "IND",
      "dates": ["2026-03-25"],
      "predicted_cases": [12345.0],
      "ci_lower": [11000.0],
      "ci_upper": [13900.0]
    }
  ]
}
```

### POST `/predict/outbreak-risk`
Request:
```json
{
  "region_ids": ["IND", "BRA", "USA"]
}
```

Response includes probability, risk level, and top factors.

### POST `/predict/scenario`
Request interventions examples:
- mobility reduction between region pairs
- vaccination acceleration in one region
- border closure simulation
- policy stringency shift

Response returns baseline, intervention forecast, and deltas.

## Cascade and Interpretability

### GET `/cascade/trace/{region_id}`
Returns probable upstream cascade chain and lag estimates.

### GET `/interpret/attention-map`
Returns weighted edges representing transmission attention corridors.

### GET `/interpret/feature-importance/{region_id}`
Returns per-feature attribution for selected region and latest snapshot.

## Error Format

All endpoints return:
```json
{
  "detail": "human readable message"
}
```
for validation or internal failures.
