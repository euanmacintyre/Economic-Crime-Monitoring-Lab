from __future__ import annotations

import pandas as pd

from econ_crime_monitoring_lab.data_gen import generate_synthetic_data
from econ_crime_monitoring_lab.features import build_transaction_features
from econ_crime_monitoring_lab.rules import score_rules


def test_rules_engine_hits_expected_rules() -> None:
    df = pd.DataFrame(
        [
            {
                "txn_id": "T1",
                "amount": 3000.0,
                "channel": "online_transfer",
                "direction": "out",
                "velocity_1h": 7,
                "small_txn_24h": 4,
                "amount_to_customer_mean": 6.0,
                "is_new_device": 1,
                "is_new_country": 1,
                "is_unusual_country": 1,
                "rapid_in_then_out": 1,
                "merchant_cash_like": 0,
            },
            {
                "txn_id": "T2",
                "amount": 20.0,
                "channel": "card_present",
                "direction": "out",
                "velocity_1h": 0,
                "small_txn_24h": 0,
                "amount_to_customer_mean": 1.0,
                "is_new_device": 0,
                "is_new_country": 0,
                "is_unusual_country": 0,
                "rapid_in_then_out": 0,
                "merchant_cash_like": 0,
            },
        ]
    )

    out = score_rules(df)

    assert out.loc[0, "rule_hit_count"] >= 5
    assert "velocity_burst" in out.loc[0, "rule_hits"]
    assert out.loc[0, "rules_risk_score"] > out.loc[1, "rules_risk_score"]
    assert out.loc[1, "rule_hits"] == "none"


def test_rules_engine_is_deterministic_for_fixed_seed() -> None:
    anchor_ts = pd.Timestamp("2025-03-01T00:00:00Z")
    bundle_one = generate_synthetic_data(
        days=5,
        seed=11,
        txns_per_day=60,
        n_customers=40,
        end_ts=anchor_ts,
    )
    bundle_two = generate_synthetic_data(
        days=5,
        seed=11,
        txns_per_day=60,
        n_customers=40,
        end_ts=anchor_ts,
    )

    features_one = build_transaction_features(
        transactions=bundle_one["transactions"],
        customers=bundle_one["customers"],
        merchants=bundle_one["merchants"],
        devices=bundle_one["devices"],
    )
    features_two = build_transaction_features(
        transactions=bundle_two["transactions"],
        customers=bundle_two["customers"],
        merchants=bundle_two["merchants"],
        devices=bundle_two["devices"],
    )

    rules_one = score_rules(features_one)[["txn_id", "rules_risk_score", "rule_hits"]]
    rules_two = score_rules(features_two)[["txn_id", "rules_risk_score", "rule_hits"]]

    assert rules_one.equals(rules_two)
