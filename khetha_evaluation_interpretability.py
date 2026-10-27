"""Evaluation, explainability, and drift monitoring for Khetha fraud models."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("khetha.evaluation")


class KhethaEvaluator:
    """Assess classification performance, attributions, and temporal drift."""

    def __init__(self, model, X_train, y_train, X_test, y_test, config=None):
        self.model = model
        self.X_train, self.y_train = X_train, np.asarray(y_train)
        self.X_test, self.y_test = X_test, np.asarray(y_test)
        self.config = config or {}
        self.y_pred = np.asarray(model.predict(X_test))
        self.y_proba = self._prediction_scores(model, X_test)

    @staticmethod
    def _prediction_scores(model, features):
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(features))
            if probabilities.ndim == 2 and probabilities.shape[1] > 1:
                return probabilities[:, 1]
        if hasattr(model, "decision_function"):
            return np.asarray(model.decision_function(features)).reshape(-1)
        return None

    @staticmethod
    def _slice_rows(values, indices):
        if hasattr(values, "iloc"):
            return values.iloc[indices]
        return np.asarray(values)[indices]

    def run_advanced_models(self, model_type):
        """Fit an SVC, SVR, or PCA-reduced Manhattan-distance KNN model."""
        normalized_type = model_type.lower().replace("-", "_")
        if normalized_type in {"svc", "svm", "svc_rbf", "svm_rbf"}:
            candidate = SVC(kernel="rbf", class_weight="balanced", random_state=42)
        elif normalized_type in {"svc_poly", "svm_poly"}:
            candidate = SVC(kernel="poly", degree=3, class_weight="balanced", random_state=42)
        elif normalized_type == "svr":
            candidate = SVR(kernel="rbf", epsilon=0.1)
        elif normalized_type in {"knn", "knn_classifier", "knn_classifier_rbf"}:
            candidate = self._make_knn_classifier()
        elif normalized_type == "knn_regressor":
            candidate = self._make_knn_regressor()
        else:
            raise ValueError("model_type must be svc, svc_poly, svr, knn_classifier, or knn_regressor")

        candidate.fit(self.X_train, self.y_train)
        self.model = candidate
        self.y_pred = np.asarray(candidate.predict(self.X_test))
        self.y_proba = self._prediction_scores(candidate, self.X_test)
        return candidate

    def _make_pca(self):
        feature_count = self.X_train.shape[1]
        sample_count = len(self.y_train)
        component_count = min(15, feature_count, sample_count)
        return PCA(n_components=component_count, random_state=42)

    def _make_knn_classifier(self):
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            self._make_pca(),
            KNeighborsClassifier(
                n_neighbors=min(5, len(self.y_train)),
                algorithm="ball_tree",
                metric="manhattan",
            ),
        )

    def _make_knn_regressor(self):
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            self._make_pca(),
            KNeighborsRegressor(
                n_neighbors=min(5, len(self.y_train)),
                algorithm="ball_tree",
                metric="manhattan",
            ),
        )

    def generate_classification_report(self):
        """Return thresholded metrics, ranking curves, and estimated costs in ZAR."""
        labels = np.unique(np.concatenate((self.y_test, self.y_pred)))
        matrix = confusion_matrix(self.y_test, self.y_pred, labels=[0, 1])
        true_negative, false_positive, false_negative, true_positive = matrix.ravel()
        metrics = {
            "precision": float(precision_score(self.y_test, self.y_pred, zero_division=0)),
            "recall": float(recall_score(self.y_test, self.y_pred, zero_division=0)),
            "f1": float(f1_score(self.y_test, self.y_pred, zero_division=0)),
            "roc_auc": None,
            "pr_auc": None,
        }
        roc_points = {"fpr": [], "tpr": [], "thresholds": []}
        pr_points = {"precision": [], "recall": [], "thresholds": []}
        if self.y_proba is not None and len(np.unique(self.y_test)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(self.y_test, self.y_proba))
            metrics["pr_auc"] = float(average_precision_score(self.y_test, self.y_proba))
            fpr, tpr, roc_thresholds = roc_curve(self.y_test, self.y_proba)
            curve_precision, curve_recall, pr_thresholds = precision_recall_curve(self.y_test, self.y_proba)
            roc_points = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thresholds.tolist()}
            pr_points = {
                "precision": curve_precision.tolist(),
                "recall": curve_recall.tolist(),
                "thresholds": pr_thresholds.tolist(),
            }

        fraud_config = self.config.get("fraud_detection", {})
        false_positive_cost = float(fraud_config.get("investigation_cost_zar", 2500))
        false_negative_cost = float(fraud_config.get("avg_fraud_loss_zar", 45000))
        business_costs = {
            "currency": "ZAR",
            "false_positives": float(false_positive * false_positive_cost),
            "false_negatives": float(false_negative * false_negative_cost),
            "false_positive_unit_cost": false_positive_cost,
            "false_negative_unit_cost": false_negative_cost,
            "total": float(false_positive * false_positive_cost + false_negative * false_negative_cost),
        }
        return {
            "labels": labels.tolist(),
            "confusion_matrix": matrix.tolist(),
            "counts": {
                "true_negative": int(true_negative),
                "false_positive": int(false_positive),
                "false_negative": int(false_negative),
                "true_positive": int(true_positive),
            },
            "metrics": metrics,
            "curves": {"roc": roc_points, "precision_recall": pr_points},
            "business_costs_zar": business_costs,
        }

    def apply_shap_explanations(self, n_samples=50, output_dir=None):
        """Save a global SHAP summary and force plots for up to three cases."""
        import shap

        output_path = Path(output_dir or ROOT / "khetha_artifacts")
        output_path.mkdir(parents=True, exist_ok=True)
        sample_count = min(max(1, int(n_samples)), len(self.X_test))
        features = self._slice_rows(self.X_test, np.arange(sample_count))
        feature_names = list(getattr(self.X_test, "columns", [f"feature_{i}" for i in range(features.shape[1])]))
        explainer = shap.TreeExplainer(self.model)
        explanation = explainer(features)
        shap_values = np.asarray(explanation.values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]
        base_values = np.asarray(explanation.base_values)
        if base_values.ndim > 1:
            base_values = base_values[:, -1]
        global_values = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(global_values)[::-1]

        plt.figure()
        shap.summary_plot(shap_values, features, feature_names=feature_names, show=False)
        summary_path = output_path / "shap_summary.png"
        plt.tight_layout()
        plt.savefig(summary_path, bbox_inches="tight")
        plt.close()
        if not summary_path.is_file() or summary_path.stat().st_size == 0:
            raise OSError(f"SHAP summary plot was not written: {summary_path}")

        local_plots = []
        for sample_index in range(min(3, sample_count)):
            baseline = base_values[sample_index] if base_values.ndim else base_values.item()
            force_plot = shap.force_plot(
                baseline,
                shap_values[sample_index],
                features.iloc[sample_index] if hasattr(features, "iloc") else features[sample_index],
                feature_names=feature_names,
            )
            force_path = output_path / f"shap_force_claim_{sample_index + 1}.html"
            shap.save_html(str(force_path), force_plot)
            if not force_path.is_file() or force_path.stat().st_size == 0:
                raise OSError(f"SHAP force plot was not written: {force_path}")
            local_plots.append(str(force_path))

        return {
            "summary_plot": str(summary_path),
            "summary_data": [
                {"feature": feature_names[index], "mean_absolute_shap": float(global_values[index])}
                for index in top_indices
            ],
            "local_force_plots": local_plots,
            "explained_samples": sample_count,
        }

    def temporal_validation(self, X_historical, y_historical, current_date_col):
        """Score expanding chronological folds and compare the latest fold to prior folds."""
        if isinstance(current_date_col, str):
            dates = pd.to_datetime(X_historical[current_date_col], errors="coerce")
            features = X_historical.drop(columns=[current_date_col])
        else:
            dates = pd.to_datetime(np.asarray(current_date_col), errors="coerce")
            features = X_historical

        date_values = np.asarray(dates)
        labels = np.asarray(y_historical)
        if len(date_values) != len(labels) or len(features) != len(labels):
            raise ValueError("Historical features, labels, and dates must have matching lengths")
        valid_indices = np.flatnonzero(~pd.isna(date_values))
        if len(valid_indices) < 5:
            raise ValueError("At least five valid dated samples are required for TimeSeriesSplit")
        features = self._slice_rows(features, valid_indices)
        labels = labels[valid_indices]
        date_values = date_values[valid_indices]
        order = np.argsort(date_values, kind="stable")
        features = self._slice_rows(features, order)
        labels = labels[order]
        sorted_dates = date_values[order]

        splitter = TimeSeriesSplit(n_splits=4)
        fold_metrics = []
        quarter_predictions = {}
        for fold_index, (train_indices, validation_indices) in enumerate(splitter.split(features), start=1):
            fold_model = clone(self.model)
            fold_model.fit(self._slice_rows(features, train_indices), labels[train_indices])
            predictions = fold_model.predict(self._slice_rows(features, validation_indices))
            validation_labels = labels[validation_indices]
            fold_f1 = float(f1_score(validation_labels, predictions, zero_division=0))
            fold_recall = float(recall_score(validation_labels, predictions, zero_division=0))
            quarter = str(pd.Timestamp(sorted_dates[validation_indices[-1]]).to_period("Q"))
            fold_metrics.append({
                "fold": fold_index,
                "quarter": quarter,
                "recall": fold_recall,
                "f1": fold_f1,
                "validation_samples": len(validation_indices),
            })
            for sample_index, prediction in zip(validation_indices, predictions):
                quarter = str(pd.Timestamp(sorted_dates[sample_index]).to_period("Q"))
                quarter_predictions.setdefault(quarter, {"actual": [], "predicted": []})
                quarter_predictions[quarter]["actual"].append(labels[sample_index])
                quarter_predictions[quarter]["predicted"].append(prediction)

        quarter_metrics = []
        for quarter in sorted(quarter_predictions, key=pd.Period):
            actual = quarter_predictions[quarter]["actual"]
            predicted = quarter_predictions[quarter]["predicted"]
            quarter_metrics.append({
                "quarter": quarter,
                "recall": float(recall_score(actual, predicted, zero_division=0)),
                "f1": float(f1_score(actual, predicted, zero_division=0)),
                "validation_samples": len(actual),
            })

        if not quarter_metrics:
            raise ValueError("No valid calendar quarters were available for evaluation")
        prior_quarters = quarter_metrics[:-1]
        prior_f1 = float(np.mean([quarter["f1"] for quarter in prior_quarters])) if prior_quarters else 0.0
        latest_f1 = float(quarter_metrics[-1]["f1"])
        relative_f1_decay = (
            max(0.0, (prior_f1 - latest_f1) / prior_f1) if prior_f1 > 0 else 0.0
        )
        return {
            "folds": fold_metrics,
            "quarters": quarter_metrics,
            "rolling_recall": [quarter["recall"] for quarter in quarter_metrics],
            "rolling_f1": [quarter["f1"] for quarter in quarter_metrics],
            "baseline_f1": prior_f1,
            "latest_f1": latest_f1,
            "relative_f1_decay": relative_f1_decay,
            "performance_drop_gt_10_percent": bool(prior_quarters) and relative_f1_decay > 0.10,
        }

    def calculate_psi_drift(self, feature_name, expected_dist, actual_dist):
        """Calculate PSI from numeric quantile bins or categorical value bins."""
        expected = np.asarray(expected_dist)
        actual = np.asarray(actual_dist)
        expected = expected[~pd.isna(expected)]
        actual = actual[~pd.isna(actual)]
        if expected.size == 0 or actual.size == 0:
            raise ValueError("Expected and actual distributions must both contain observations")

        if np.issubdtype(expected.dtype, np.number) and np.issubdtype(actual.dtype, np.number):
            edges = np.unique(np.quantile(expected.astype(float), np.linspace(0, 1, 11)))
            if edges.size < 2:
                center = float(edges[0])
                half_width = max(abs(center) * 1e-6, 1e-6)
                edges = np.array([-np.inf, center - half_width, center + half_width, np.inf])
            else:
                edges[0], edges[-1] = -np.inf, np.inf
            expected_counts = np.histogram(expected.astype(float), bins=edges)[0]
            actual_counts = np.histogram(actual.astype(float), bins=edges)[0]
        else:
            categories = np.unique(np.concatenate((expected.astype(str), actual.astype(str))))
            expected_counts = np.array([np.sum(expected.astype(str) == category) for category in categories])
            actual_counts = np.array([np.sum(actual.astype(str) == category) for category in categories])

        epsilon = 1e-6
        expected_percentages = expected_counts.astype(float) / expected_counts.sum()
        actual_percentages = actual_counts.astype(float) / actual_counts.sum()
        expected_safe = np.clip(expected_percentages, epsilon, None)
        actual_safe = np.clip(actual_percentages, epsilon, None)
        psi = float(np.sum((actual_safe - expected_safe) * np.log(actual_safe / expected_safe)))
        retrain_threshold = float(self.config.get("model_retraining", {}).get("psi_drift_threshold", 0.15))
        drift_level = "stable" if psi < 0.10 else "monitor" if psi < retrain_threshold else "retrain"
        return {
            "feature": feature_name,
            "psi": psi,
            "drift_level": drift_level,
            "expected_percentages": expected_percentages.tolist(),
            "actual_percentages": actual_percentages.tolist(),
            "retrain_threshold": retrain_threshold,
        }

    def should_retrain(self, psi_results, temporal_results):
        """Apply configured PSI and rolling-F1 thresholds to request retraining."""
        psi_threshold = float(self.config.get("model_retraining", {}).get("psi_drift_threshold", 0.15))
        f1_threshold = float(self.config.get("model_retraining", {}).get("f1_decay_threshold", 0.05))
        drifted_features = [result["feature"] for result in psi_results if result["psi"] >= psi_threshold]
        f1_decay = float(temporal_results.get("relative_f1_decay", 0.0))
        return {
            "retrain": bool(drifted_features or f1_decay >= f1_threshold),
            "reasons": {
                "psi_threshold_exceeded": drifted_features,
                "rolling_f1_decay_exceeded": f1_decay >= f1_threshold,
            },
        }


def _load_config():
    config_path = ROOT / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as config_file:
        return json.load(config_file)


def _prepare_claim_data():
    claims_path = ROOT / "claims_data.csv"
    claims = pd.read_csv(claims_path, parse_dates=["claim_date"])
    claims = (
        claims.dropna(subset=["fraud_label", "claim_date"])
        .sort_values("claim_date", kind="stable")
        .reset_index(drop=True)
    )
    if claims.empty:
        raise ValueError("No valid claims with labels and dates were found")
    target = claims["fraud_label"].astype(int).to_numpy()
    dates = claims["claim_date"].copy()
    excluded = {
        "claim_id", "fraud_label", "investigation_outcome", "claim_settlement_date", "claim_date"
    }
    raw_features = claims.drop(columns=[column for column in excluded if column in claims.columns])
    split_index = int(len(raw_features) * 0.8)
    if split_index < 5 or len(raw_features) - split_index < 1:
        raise ValueError("claims_data.csv does not contain enough labelled rows to evaluate")

    train_raw = raw_features.iloc[:split_index].copy()
    test_raw = raw_features.iloc[split_index:].copy()
    categorical = train_raw.select_dtypes(
        include=["object", "string", "bool", "category"]
    ).columns.tolist()
    X_train = pd.get_dummies(train_raw, columns=categorical, dummy_na=True, dtype=float)
    X_test = pd.get_dummies(test_raw, columns=categorical, dummy_na=True, dtype=float)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    X_train = X_train.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = X_train.median().fillna(0.0)
    X_train = X_train.fillna(medians).fillna(0.0)
    X_test = X_test.fillna(medians).fillna(0.0)
    features = pd.concat([X_train, X_test], axis=0, ignore_index=True).astype(float)
    return features, target, dates, split_index


def _fit_milestone3_model(X_train, y_train, X_test, y_test):
    """Use the Milestone 3 optimizer when its module is available."""
    try:
        from khetha_ensemble_optimizer import KhethaEnsembleOptimizer
    except ImportError:
        LOGGER.warning("Milestone 3 optimizer not found; using the local random-forest fallback")
        model = RandomForestClassifier(
            n_estimators=100, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
        )
        return model.fit(X_train, y_train)

    optimizer = KhethaEnsembleOptimizer(X_train, y_train, X_test, y_test)
    optimizer.build_ensembles("random_forest")
    optimizer.run_grid_search(
        {
            "n_estimators": [100],
            "max_features": ["sqrt"],
            "min_samples_leaf": [1, 2],
            "class_weight": ["balanced"],
        },
        cv=3,
    )
    return optimizer.tuned_model


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = _load_config()
    features, target, dates, split_index = _prepare_claim_data()
    X_train, X_test = features.iloc[:split_index], features.iloc[split_index:]
    y_train, y_test = target[:split_index], target[split_index:]
    model = _fit_milestone3_model(X_train, y_train, X_test, y_test)
    evaluator = KhethaEvaluator(model, X_train, y_train, X_test, y_test, config=config)

    report = evaluator.generate_classification_report()
    LOGGER.info("Holdout metrics: %s", report["metrics"])
    LOGGER.info("Estimated business costs: %s", report["business_costs_zar"])

    temporal = evaluator.temporal_validation(features, target, dates)
    psi_feature = "claim_amount" if "claim_amount" in X_train else features.columns[0]
    psi_result = evaluator.calculate_psi_drift(
        psi_feature, X_train[psi_feature].to_numpy(), X_test[psi_feature].to_numpy()
    )
    retraining = evaluator.should_retrain([psi_result], temporal)
    LOGGER.info("Temporal validation: %s", temporal)
    LOGGER.info("Feature drift: %s", psi_result)
    LOGGER.info("Retraining decision: %s", retraining)

    explanations = evaluator.apply_shap_explanations(n_samples=50)
    LOGGER.info("SHAP artifacts: %s", explanations)


if __name__ == "__main__":
    main()