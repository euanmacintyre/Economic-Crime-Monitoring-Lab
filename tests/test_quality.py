from __future__ import annotations

import pandas as pd
import pytest

from econ_crime_monitoring_lab.quality import DataQualityError, validate_transactions


def _valid_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "txn_id": "T0001",
                "timestamp": pd.Timestamp("2025-01-01T10:00:00Z"),
                "amount": 42.5,
                "currency": "USD",
                "customer_id": "C000001",
                "account_id": "A000001",
                "merchant_id": "M000001",
                "channel": "card_present",
                "country": "US",
                "device_id": "D000001",
                "beneficiary_account": None,
                "direction": "out",
                "typology_label": "none",
                "is_suspicious": 0,
            }
        ]
    )


def test_validate_transactions_passes_on_valid_data() -> None:
    df = _valid_transactions()
    validate_transactions(df)


def test_validate_transactions_fails_on_negative_amount() -> None:
    df = _valid_transactions()
    df.loc[0, "amount"] = -5

    with pytest.raises(DataQualityError, match="Non-positive amounts"):
        validate_transactions(df)


def test_validate_transactions_fails_on_transfer_without_beneficiary() -> None:
    df = _valid_transactions()
    df.loc[0, "channel"] = "online_transfer"
    df.loc[0, "beneficiary_account"] = None

    with pytest.raises(DataQualityError, match="Missing beneficiary_account"):
        validate_transactions(df)
