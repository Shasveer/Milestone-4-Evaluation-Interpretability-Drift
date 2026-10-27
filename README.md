# Milestone 4: Evaluation, Interpretability, and Drift

This repository evaluates the Khetha fraud classifier beyond accuracy. The
evaluator reports classification and business-cost metrics, creates SHAP
explanations, validates performance over chronological folds, and monitors
feature drift with Population Stability Index (PSI).

## Repository Contents

- `khetha_evaluation_interpretability.py`: `KhethaEvaluator` and executable
	evaluation workflow.
- `khetha_ensemble_optimizer.py`: included Milestone 3 optimizer module from
	[the upstream repository](https://github.com/Shasveer/Milestone-3-Ensemble-Optimisation-Nested-Tuning).
	The evaluator uses its grid-searched Random Forest; a local Random Forest
	fallback keeps the workflow runnable if the module is unavailable.
- `claims_data.csv`: claim-level model features, dates, and `fraud_label` target.
- `fraud_labels.csv`, `vehicles.csv`, `policyholders.csv`, and
	`geographic_risk.csv`: accompanying investigation and domain datasets.
- `config.json`: fraud costs and monitoring thresholds.
- `requirements.txt`: Python runtime dependencies.

The supplied geographic file is named `geographic_risk.csv` (singular).
`geographic_risks.csv` is also present as an alias for submission checklists
that use the plural spelling. The evaluator does not require joining the file
because the claim table already includes the fields needed for this milestone.

## Setup and Run

Use Python 3.12 or newer. The exact dependency versions validated for this
submission are pinned in `requirements.txt`. From the repository root:

```bash
python -m pip install -r requirements.txt
python khetha_evaluation_interpretability.py
```

The script sorts claims by `claim_date`, uses the first 80% for training and
the most recent 20% as a holdout. Categorical columns and numeric imputation
medians are learned from the training period only, then applied to the holdout
with the same feature schema. Identifiers and post-claim investigation
outcomes are excluded from model features. It then fits the Milestone 3
optimizer, prints evaluation and monitoring summaries, and writes a global
SHAP summary image plus three local force-plot HTML files to
`khetha_artifacts/`. SHAP failures stop the run rather than being reported as a
successful evaluation.

## Evaluator Capabilities

- `run_advanced_models()` supports RBF and polynomial SVC, epsilon-tube SVR,
	and KNN classification/regression with a Ball Tree, Manhattan distance,
	scaling, and up to 15 PCA components.
- `generate_classification_report()` returns the confusion matrix, precision,
	recall, F1, ROC-AUC, precision-recall AUC, curve coordinates, and estimated
	false-positive and false-negative costs in ZAR. Costs default to the values
	in `config.json`.
- `apply_shap_explanations()` uses `shap.TreeExplainer`, returns global
	mean-absolute attribution data, and saves local force plots for up to three
	holdout claims.
- `temporal_validation()` uses `TimeSeriesSplit(n_splits=4)` on ordered data
	and reports fold-level plus calendar-quarter recall/F1. The latest quarter's
	F1 is compared with the mean of earlier evaluated quarters and flagged when
	it falls by more than 10%. Invalid dates are excluded, and a one-quarter
	history does not produce a decline alert without a historical baseline.
- `calculate_psi_drift()` bins numerical features at expected-distribution
	quantiles (or uses category bins), smooths empty bins, and reports PSI as
	stable, monitor, or retrain. `should_retrain()` combines the configured PSI
	threshold with rolling F1 decay.

The public methods accept compatible NumPy arrays and pandas objects, making
the evaluator usable independently of the bundled command-line workflow.