from __future__ import annotations

from econ_crime_monitoring_lab.data_gen import generate_synthetic_data
from econ_crime_monitoring_lab.features import build_transaction_features
from econ_crime_monitoring_lab.model import train_and_score_model
from econ_crime_monitoring_lab.rules import score_rules


def test_model_training_pipeline_runs() -> None:
    bundle = generate_synthetic_data(days=14, seed=7, txns_per_day=90, n_customers=120)
    featured = build_transaction_features(
        transactions=bundle["transactions"],
        customers=bundle["customers"],
        merchants=bundle["merchants"],
        devices=bundle["devices"],
    )
    ruled = score_rules(featured)

    artifacts = train_and_score_model(ruled)

    assert "auc" in artifacts.metrics
    assert "precision" in artifacts.metrics
    assert "recall" in artifacts.metrics
    assert len(artifacts.scored_transactions) == len(ruled)
    assert artifacts.top_features.shape[0] > 0

