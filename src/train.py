"""
Model training and pipeline orchestration module.
Defines classification pipelines (Logistic Regression, Decision Tree, Random Forest,
Gradient Boosting), performs cross-validation, hyperparameter tuning, model selection,
and persists the final pipeline and metadata with Joblib.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import DEFAULT_DATA_PATH, load_data
from src.evaluate import evaluate_classifier
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_preprocessor,
    prepare_train_test_data,
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
ALL_EVALUATIONS_PATH = os.path.join(MODELS_DIR, "all_evaluations.json")


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns candidate classifiers with appropriate imbalance handling and regularizations.
    """
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=25,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=3,
            random_state=random_state,
        ),
    }


def build_pipeline_for_model(classifier: Any) -> Pipeline:
    """
    Combines preprocessing ColumnTransformer and classifier into a unified Pipeline.
    """
    preprocessor = build_preprocessor()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def run_cross_validation(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Performs Stratified K-Fold Cross-Validation on the training data only.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    return {
        "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
        "cv_accuracy_std": float(np.std(scores["test_accuracy"])),
        "cv_precision_mean": float(np.mean(scores["test_precision"])),
        "cv_precision_std": float(np.std(scores["test_precision"])),
        "cv_recall_mean": float(np.mean(scores["test_recall"])),
        "cv_recall_std": float(np.std(scores["test_recall"])),
        "cv_f1_mean": float(np.mean(scores["test_f1"])),
        "cv_f1_std": float(np.std(scores["test_f1"])),
        "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(scores["test_roc_auc"])),
    }


def train_and_compare_models(
    df: Optional[pd.DataFrame] = None,
    random_state: int = 42,
    cv_folds: int = 5
) -> Tuple[str, Pipeline, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Trains all candidate pipelines on training data, evaluates on test data and CV,
    selects the optimal model based on Telco business objectives (high Recall & F1 while preserving ROC-AUC),
    and returns (best_name, best_pipeline, all_evaluations, metadata).
    """
    if df is None:
        df = load_data()

    X_train, X_test, y_train, y_test = prepare_train_test_data(df, test_size=0.2, random_state=random_state)

    candidate_models = get_candidate_models(random_state=random_state)
    all_evaluations: List[Dict[str, Any]] = []
    trained_pipelines: Dict[str, Pipeline] = {}

    for name, classifier in candidate_models.items():
        pipeline = build_pipeline_for_model(classifier)

        # 1. Stratified Cross-Validation on training set
        cv_metrics = run_cross_validation(
            pipeline, X_train, y_train, cv_folds=cv_folds, random_state=random_state
        )

        # 2. Fit pipeline on full training set
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # 3. Evaluate on held-out test set
        test_eval = evaluate_classifier(pipeline, X_test, y_test, model_name=name)
        test_eval.update(cv_metrics)
        all_evaluations.append(test_eval)

    # 4. Model Selection Strategy:
    # Business context: A False Negative (lost customer) costs significantly more than a False Positive (retention incentive).
    # We rank candidates by F1-score and Recall, penalized slightly for poor Precision or ROC-AUC.
    # Composite selection metric: 0.40 * F1 + 0.35 * Recall + 0.25 * ROC-AUC (all test or CV)
    best_name = None
    best_score = -1.0
    for eval_item in all_evaluations:
        f1 = eval_item["f1_score"]
        rec = eval_item["recall"]
        auc_val = eval_item["roc_auc"] or 0.5
        composite = 0.40 * f1 + 0.35 * rec + 0.25 * auc_val
        eval_item["composite_selection_score"] = float(composite)
        if composite > best_score:
            best_score = composite
            best_name = eval_item["model_name"]

    best_pipeline = trained_pipelines[best_name]
    best_eval = next(item for item in all_evaluations if item["model_name"] == best_name)

    metadata = {
        "model_name": best_name,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "random_state": random_state,
        "cv_folds": cv_folds,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "selection_rationale": (
            f"Selected '{best_name}' using composite business score ({best_score:.3f}) balancing "
            f"Recall ({best_eval['recall']:.3f}), F1-Score ({best_eval['f1_score']:.3f}), and "
            f"ROC-AUC ({best_eval['roc_auc']:.3f}). Prioritizes early churn detection to minimize customer loss."
        ),
        "metrics": {
            "accuracy": best_eval["accuracy"],
            "precision": best_eval["precision"],
            "recall": best_eval["recall"],
            "f1_score": best_eval["f1_score"],
            "roc_auc": best_eval["roc_auc"],
            "cv_f1_mean": best_eval.get("cv_f1_mean"),
            "cv_recall_mean": best_eval.get("cv_recall_mean"),
            "cv_roc_auc_mean": best_eval.get("cv_roc_auc_mean"),
        },
        "model_params": {
            k: str(v) for k, v in best_pipeline.named_steps["classifier"].get_params().items()
        },
    }

    return best_name, best_pipeline, all_evaluations, metadata


def save_trained_artifacts(
    best_pipeline: Pipeline,
    metadata: Dict[str, Any],
    all_evaluations: List[Dict[str, Any]],
    model_path: str = BEST_MODEL_PATH,
    metadata_path: str = METADATA_PATH,
    all_evals_path: str = ALL_EVALUATIONS_PATH,
) -> None:
    """
    Persists the best pipeline and evaluation metadata safely to disk.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_pipeline, model_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(all_evals_path, "w", encoding="utf-8") as f:
        json.dump(all_evaluations, f, indent=2)


def load_best_model(model_path: str = BEST_MODEL_PATH) -> Optional[Pipeline]:
    """
    Loads saved model pipeline.
    Documents that joblib files should only be loaded from trusted local sources.
    """
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def load_model_metadata(metadata_path: str = METADATA_PATH) -> Optional[Dict[str, Any]]:
    """
    Loads saved training metadata.
    """
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_evaluations(eval_path: str = ALL_EVALUATIONS_PATH) -> Optional[List[Dict[str, Any]]]:
    """
    Loads saved multi-model evaluation data.
    """
    if not os.path.exists(eval_path):
        return None
    with open(eval_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    print("Starting Telco Customer Churn model training workflow...")
    best_name, best_pipeline, all_evals, metadata = train_and_compare_models()
    save_trained_artifacts(best_pipeline, metadata, all_evals)
    print(f"\nTraining complete. Best model: {best_name}")
    print(f"Selection Rationale: {metadata['selection_rationale']}")
    print(f"Artifacts saved to {MODELS_DIR}")
