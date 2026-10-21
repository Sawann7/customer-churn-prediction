"""
Unit tests for model training, cross-validation, and artifact persistence.
"""

import os
import tempfile
import pandas as pd
import pytest
from src.data_loader import generate_demo_dataset
from src.evaluate import evaluate_classifier
from src.train import (
    build_pipeline_for_model,
    get_candidate_models,
    load_best_model,
    load_model_metadata,
    run_cross_validation,
    save_trained_artifacts,
    train_and_compare_models,
)


def test_candidate_models_initialization():
    candidates = get_candidate_models(random_state=42)
    assert "Logistic Regression" in candidates
    assert "Decision Tree" in candidates
    assert "Random Forest" in candidates
    assert "Gradient Boosting" in candidates


def test_pipeline_build_and_fit():
    demo_df = generate_demo_dataset(80, random_state=42)
    from src.preprocessing import prepare_train_test_data

    X_train, X_test, y_train, y_test = prepare_train_test_data(demo_df, test_size=0.25, random_state=42)

    candidates = get_candidate_models(random_state=42)
    lr = candidates["Logistic Regression"]
    pipeline = build_pipeline_for_model(lr)

    pipeline.fit(X_train, y_train)
    eval_result = evaluate_classifier(pipeline, X_test, y_test, model_name="Logistic Regression")

    assert 0.0 <= eval_result["accuracy"] <= 1.0
    assert 0.0 <= eval_result["f1_score"] <= 1.0
    assert eval_result["roc_auc"] is not None
    assert 0.0 <= eval_result["roc_auc"] <= 1.0


def test_run_cross_validation():
    demo_df = generate_demo_dataset(60, random_state=42)
    from src.preprocessing import prepare_train_test_data

    X_train, _, y_train, _ = prepare_train_test_data(demo_df, test_size=0.2, random_state=42)

    candidates = get_candidate_models(random_state=42)
    pipeline = build_pipeline_for_model(candidates["Decision Tree"])

    cv_scores = run_cross_validation(pipeline, X_train, y_train, cv_folds=3, random_state=42)

    assert "cv_f1_mean" in cv_scores
    assert "cv_roc_auc_mean" in cv_scores
    assert 0.0 <= cv_scores["cv_f1_mean"] <= 1.0


def test_save_and_load_artifacts():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.joblib")
        meta_path = os.path.join(tmpdir, "test_metadata.json")
        eval_path = os.path.join(tmpdir, "test_evals.json")

        demo_df = generate_demo_dataset(50, random_state=42)
        best_name, best_pipeline, all_evals, metadata = train_and_compare_models(
            df=demo_df, random_state=42, cv_folds=2
        )

        save_trained_artifacts(
            best_pipeline=best_pipeline,
            metadata=metadata,
            all_evaluations=all_evals,
            model_path=model_path,
            metadata_path=meta_path,
            all_evals_path=eval_path,
        )

        assert os.path.exists(model_path)
        assert os.path.exists(meta_path)
        assert os.path.exists(eval_path)

        loaded_model = load_best_model(model_path)
        assert loaded_model is not None
        assert hasattr(loaded_model, "predict")

        loaded_meta = load_model_metadata(meta_path)
        assert loaded_meta["model_name"] == best_name
