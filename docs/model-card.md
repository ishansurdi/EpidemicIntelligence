# Model Card (Draft)

## Model Name
ORACLE Epidemic Cascade Predictor

## Intended Use
- Early outbreak risk detection
- Short-term case forecasting
- Scenario-based policy exploration

## Not Intended For
- Individual medical diagnosis
- Single-source policy decisions without local epidemiological review

## Inputs
- case trends (JHU)
- vaccination, testing, policy, hospitalization (OWID)
- mobility signals (Google/OWID)

## Outputs
- 7/14/28-day case forecasts
- outbreak probability scores
- explainability artifacts (attention corridors, feature attributions)

## Performance Targets
- MAE/RMSE/MAPE for forecasting
- AUC-ROC/F1 for outbreak classification
- spatial holdout generalization on unseen countries

## Key Assumptions
- reporting quality differs by country and over time
- mobility proxies transmission opportunity, not direct infection probability
- policy indices are noisy and lag behavior changes

## Known Risks
- under-reporting can bias labels and derived features
- sudden variant shifts can degrade model calibration
- geopolitical disruptions can alter mobility topology rapidly

## Responsible Use
- always show uncertainty intervals and confidence qualifiers
- pair model output with local surveillance intelligence
- log model version, data cutoff date, and feature configuration per run

## Update Policy
- periodic retraining with drift checks
- threshold recalibration after major wave regime changes
- publish changelog for architecture and feature updates
