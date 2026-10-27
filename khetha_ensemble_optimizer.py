"""Ensemble models and leakage-safe hyperparameter optimisation for fraud data."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

try:
    import numpy as np
    from sklearn.base import clone
    from sklearn.ensemble import (
        AdaBoostClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.model_selection import (
        GridSearchCV,
        RandomizedSearchCV,
        StratifiedKFold,
        cross_validate,
        cross_val_score,
    )
    from sklearn.tree import DecisionTreeClassifier
except ImportError:  # Keep the module executable before optional dependencies are installed.
    np = None
    clone = None
    AdaBoostClassifier = GradientBoostingClassifier = RandomForestClassifier = None
    GridSearchCV = RandomizedSearchCV = StratifiedKFold = cross_val_score = None
    DecisionTreeClassifier = None


class KhethaEnsembleOptimizer:
    """Build, tune, and evaluate ensemble classifiers without test-set leakage."""

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.tuned_model = None
        self.ensemble_models = {}
        self.convergence_trajectory = []
        self.outer_scores = []

    def _require_dependencies(self):
        if np is None or RandomForestClassifier is None:
            raise ImportError("numpy and scikit-learn are required for model optimisation")

    @staticmethod
    def _take_rows(values, indices):
        if hasattr(values, "iloc"):
            return values.iloc[indices]
        if hasattr(values, "tocsr"):
            return values[indices]
        return np.asarray(values)[indices]

    def build_ensembles(self, strategy="random_forest"):
        """Initialise Random Forest, Gradient Boosting, or AdaBoost.

        ``auto`` applies a simple bias/variance rule after comparing a shallow tree,
        a bagged forest, and a boosted model with the training data only.
        """
        self._require_dependencies()
        strategy = strategy.lower()
        if strategy == "auto":
            strategy = self.diagnose_bias_variance()

        random_forest = RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            oob_score=True,
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        )
        gradient_boosting = GradientBoostingClassifier(
            learning_rate=0.05,
            n_estimators=200,
            subsample=0.8,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        )
        stump = DecisionTreeClassifier(max_depth=1, random_state=42)
        try:
            adaboost = AdaBoostClassifier(
                base_estimator=stump,
                n_estimators=200,
                learning_rate=0.05,
                random_state=42,
            )
        except TypeError:  # scikit-learn >= 1.2 renamed base_estimator to estimator.
            adaboost = AdaBoostClassifier(
                estimator=stump,
                n_estimators=200,
                learning_rate=0.05,
                random_state=42,
            )

        self.ensemble_models = {
            "random_forest": random_forest,
            "gradient_boosting": gradient_boosting,
            "adaboost": adaboost,
        }
        aliases = {"rf": "random_forest", "gbm": "gradient_boosting", "gb": "gradient_boosting", "ada": "adaboost"}
        selected = aliases.get(strategy, strategy)
        if selected not in self.ensemble_models:
            raise ValueError("strategy must be random_forest, gradient_boosting, adaboost, or auto")
        self.selected_strategy = selected
        self.tuned_model = self.ensemble_models[selected]
        return self.tuned_model

    def diagnose_bias_variance(self, cv=3):
        """Measure fit gaps, OOB performance, and noise before choosing a strategy."""
        self._require_dependencies()
        tree = DecisionTreeClassifier(max_depth=3, random_state=42)
        forest = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42, n_jobs=-1)
        boosted = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        diagnostics = {}
        for name, model in (("tree", tree), ("random_forest", forest), ("gradient_boosting", boosted)):
            measured = cross_validate(
                model, self.X_train, self.y_train, cv=folds, scoring="roc_auc",
                return_train_score=True, n_jobs=-1,
            )
            model.fit(self.X_train, self.y_train)
            train_score = float(np.mean(measured["train_score"]))
            validation_score = float(np.mean(measured["test_score"]))
            diagnostics[name] = {
                "train_score": train_score,
                "validation_score": validation_score,
                "generalisation_gap": train_score - validation_score,
                "validation_std": float(np.std(measured["test_score"])),
            }
            if name == "random_forest":
                diagnostics[name]["oob_score"] = float(model.oob_score_)

        tree_profile = diagnostics["tree"]
        forest_profile = diagnostics["random_forest"]
        boosted_profile = diagnostics["gradient_boosting"]
        noise_detected = (
            forest_profile["generalisation_gap"] > 0.15
            or forest_profile["validation_std"] > 0.10
            or forest_profile["oob_score"] + 0.10 < forest_profile["validation_score"]
        )
        diagnostics["noise_detected"] = bool(noise_detected)
        self.bias_variance_profile = diagnostics
        high_variance = tree_profile["generalisation_gap"] > 0.10
        high_bias = tree_profile["train_score"] < 0.80
        if noise_detected:
            self.bias_variance_strategy_reason = "noise detected; prefer conservative bagging"
            return "random_forest"
        if high_variance and forest_profile["validation_score"] >= tree_profile["validation_score"]:
            self.bias_variance_strategy_reason = "high variance; prefer bagging"
            return "random_forest"
        if high_bias and boosted_profile["validation_score"] > tree_profile["validation_score"] + 0.01:
            self.bias_variance_strategy_reason = "high bias with boosting improvement"
            return "gradient_boosting"
        self.bias_variance_strategy_reason = "no decisive improvement; prefer conservative bagging"
        return "random_forest"

    def monitor_gradient_boosting(self, validation_fraction=0.2):
        """Use staged predictions to select the best validation iteration."""
        self._require_dependencies()
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        X_fit, X_valid, y_fit, y_valid = train_test_split(
            self.X_train, self.y_train, test_size=validation_fraction,
            stratify=self.y_train, random_state=42,
        )
        configured_model = self.ensemble_models.get("gradient_boosting")
        if configured_model is None:
            configured_model = self.build_ensembles("gradient_boosting")
        model = clone(configured_model)
        model.set_params(n_iter_no_change=None)
        model.fit(X_fit, y_fit)
        validation_scores = [accuracy_score(y_valid, prediction) for prediction in model.staged_predict(X_valid)]
        self.gradient_validation_scores = validation_scores
        self.best_gradient_stage = int(np.argmax(validation_scores)) + 1
        return self.best_gradient_stage, max(validation_scores)

    def track_adaboost_weights(self):
        """Fit AdaBoost and expose its per-round learner weight trajectory."""
        self._require_dependencies()
        configured_model = self.ensemble_models.get("adaboost")
        if configured_model is None:
            configured_model = self.build_ensembles("adaboost")
        model = clone(configured_model)
        model.fit(self.X_train, self.y_train)
        self.adaboost_weight_trajectory = list(getattr(model, "estimator_weights_", []))
        return self.adaboost_weight_trajectory

    def run_grid_search(self, param_grid, cv=5):
        """Exhaustively tune the selected estimator on the training data only."""
        self._require_dependencies()
        if self.tuned_model is None:
            self.build_ensembles()
        search = GridSearchCV(self.tuned_model, param_grid=param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
        search.fit(self.X_train, self.y_train)
        self.tuned_model = search.best_estimator_
        return {"best_params": search.best_params_, "best_score": search.best_score_, "cv_results": search.cv_results_, "search": search}

    def run_randomized_search(self, param_distributions=None, n_iter=50, cv=5):
        """Sample broad scipy distributions and retain the score trajectory."""
        self._require_dependencies()
        from scipy.stats import loguniform, randint

        if self.tuned_model is None:
            self.build_ensembles()
        if param_distributions is None:
            if isinstance(self.tuned_model, GradientBoostingClassifier):
                param_distributions = {
                    "n_estimators": randint(50, 301),
                    "learning_rate": loguniform(0.01, 0.3),
                    "max_depth": randint(1, 6),
                }
            else:
                param_distributions = {
                    "n_estimators": randint(50, 301),
                    "max_features": ["sqrt", "log2", None],
                }
        search = RandomizedSearchCV(
            self.tuned_model, param_distributions=param_distributions, n_iter=n_iter,
            cv=cv, scoring="roc_auc", n_jobs=-1, random_state=42, refit=True,
        )
        search.fit(self.X_train, self.y_train)
        self.tuned_model = search.best_estimator_
        self.convergence_trajectory = list(np.maximum.accumulate(search.cv_results_["mean_test_score"]))
        return {"best_params": search.best_params_, "best_score": search.best_score_, "cv_results": search.cv_results_, "convergence_trajectory": self.convergence_trajectory, "search": search, "loguniform": loguniform}

    def run_bayesian_optimization(self, n_trials=40):
        """Tune a forest with Optuna's TPE sampler using only inner CV scores."""
        self._require_dependencies()
        import optuna

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        def objective(trial):
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 2, 20),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                oob_score=True, bootstrap=True, random_state=42, n_jobs=-1,
            )
            return float(np.mean(cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring="roc_auc", n_jobs=-1)))

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        self.tuned_model = RandomForestClassifier(**study.best_params, oob_score=True, bootstrap=True, random_state=42, n_jobs=-1)
        self.bayesian_study = study
        return {"best_params": study.best_params, "best_score": study.best_value, "study": study, "objective": objective}

    def nested_cross_validation(self, outer_cv=5, inner_cv=3):
        """Tune in each outer training fold, then score once on its untouched fold."""
        self._require_dependencies()
        outer = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42)
        scores = []
        grid = {"n_estimators": [100, 200], "max_features": ["sqrt", "log2"], "min_samples_leaf": [1, 2]}
        for train_indices, validation_indices in outer.split(self.X_train, self.y_train):
            model = RandomForestClassifier(oob_score=True, bootstrap=True, random_state=42, n_jobs=-1)
            inner_search = GridSearchCV(model, grid, cv=inner_cv, scoring="roc_auc", n_jobs=-1)
            X_outer_train = self._take_rows(self.X_train, train_indices)
            y_outer_train = self._take_rows(self.y_train, train_indices)
            X_outer_valid = self._take_rows(self.X_train, validation_indices)
            y_outer_valid = self._take_rows(self.y_train, validation_indices)
            inner_search.fit(X_outer_train, y_outer_train)
            scores.append(float(inner_search.score(X_outer_valid, y_outer_valid)))
        scores = np.asarray(scores)
        self.outer_scores = list(scores)
        mean_score = float(np.mean(scores))
        margin = float(1.96 * np.std(scores, ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else 0.0
        return {"mean_score": mean_score, "ci_95": (max(0.0, mean_score - margin), min(1.0, mean_score + margin)), "outer_scores": self.outer_scores}


def _load_demo_data(data_dir):
    """Fallback loader used only when the earlier milestone modules are unavailable."""
    import pandas as pd

    claims = pd.read_csv(Path(data_dir) / "claims_data.csv")
    features = claims.select_dtypes(include="number").drop(columns=["fraud_label"], errors="ignore").fillna(0)
    target = claims["fraud_label"].astype(int).to_numpy()
    split = int(len(features) * 0.8)
    return features.to_numpy()[:split], target[:split], features.to_numpy()[split:], target[split:]


def _import_milestone_outputs():
    """Import the earlier milestone modules used by the full preparation path."""
    outputs = {}
    for module_name in ("khetha_data_pipeline", "khetha_baseline_models"):
        try:
            outputs[module_name] = importlib.import_module(module_name)
        except ImportError:
            continue
    return outputs


def _load_prepared_data(data_dir):
    """Run Milestones 1 and 2, returning encoded train/test arrays and labels."""
    import pandas as pd

    modules = _import_milestone_outputs()
    pipeline_module = modules.get("khetha_data_pipeline")
    baseline_module = modules.get("khetha_baseline_models")
    if pipeline_module is None or baseline_module is None:
        raise ImportError("Milestone 1 and 2 modules are required for prepared data")

    claims = pd.read_csv(Path(data_dir) / "claims_data.csv")
    pipeline = pipeline_module.KhethaDataPipeline(claims)
    X_train, X_test, y_train, y_test = pipeline.run(
        "fraud_label", ["province", "claim_type", "claim_amount"]
    )
    preprocessor = baseline_module._make_preprocessor(X_train)
    X_train_encoded = preprocessor.fit_transform(X_train, y_train)
    X_test_encoded = preprocessor.transform(X_test)
    return X_train_encoded, y_train.to_numpy(), X_test_encoded, y_test.to_numpy()


def main():
    if np is None:
        print("Install numpy, pandas, scipy, scikit-learn, and optuna to run the ensemble demo.")
        return
    try:
        data_dir = Path(__file__).parent
        try:
            X_train, y_train, X_test, y_test = _load_prepared_data(data_dir)
            print("Using Milestone 1 data preparation and Milestone 2 preprocessing.")
        except (ImportError, FileNotFoundError, KeyError, ValueError) as error:
            print(f"Prepared milestone workflow unavailable ({error}); using fallback data.")
            X_train, y_train, X_test, y_test = _load_demo_data(data_dir)

        # Keep the smoke workflow fast while preserving a completely untouched holdout.
        if len(y_train) > 1200:
            from sklearn.model_selection import train_test_split
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train, train_size=1200, stratify=y_train, random_state=42
            )
        optimizer = KhethaEnsembleOptimizer(X_train, y_train, X_test, y_test)
        optimizer.build_ensembles("random_forest")
        print("Bias/variance profile:", optimizer.diagnose_bias_variance(cv=3))
        optimizer.build_ensembles("gradient_boosting")
        print("Gradient staged validation:", optimizer.monitor_gradient_boosting())
        optimizer.build_ensembles("adaboost")
        print("AdaBoost weight rounds:", len(optimizer.track_adaboost_weights()))
        optimizer.build_ensembles("random_forest")
        print("Grid search:", optimizer.run_grid_search({"n_estimators": [20, 40], "max_features": ["sqrt", "log2"]}, cv=3)["best_params"])
        print("Randomized search:", optimizer.run_randomized_search(n_iter=2, cv=3)["convergence_trajectory"])
        print("Bayesian search:", optimizer.run_bayesian_optimization(n_trials=2)["best_params"])
        print("Nested CV:", optimizer.nested_cross_validation(outer_cv=3, inner_cv=2))
    except (ImportError, ValueError) as error:
        print(f"Ensemble demo unavailable: {error}")


if __name__ == "__main__":
    main()