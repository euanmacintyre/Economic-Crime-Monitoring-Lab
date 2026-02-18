from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator


REQUIRED_TXN_COLUMNS = [
    "txn_id",
    "timestamp",
    "amount",
    "currency",
    "customer_id",
    "account_id",
    "merchant_id",
    "channel",
    "country",
    "device_id",
    "beneficiary_account",
    "direction",
    "typology_label",
    "is_suspicious",
]


class TransactionRecord(BaseModel):
    txn_id: str
    timestamp: datetime
    amount: float = Field(gt=0)
    currency: str
    customer_id: str
    account_id: str
    merchant_id: str
    channel: str
    country: str
    device_id: str
    beneficiary_account: str | None = None
    direction: str
    typology_label: str
    is_suspicious: int = Field(ge=0, le=1)

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        allowed = {"USD", "EUR", "GBP"}
        if value not in allowed:
            raise ValueError(f"currency must be one of {sorted(allowed)}")
        return value

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, value: str) -> str:
        allowed = {"card_present", "card_not_present", "online_transfer", "atm_cash"}
        if value not in allowed:
            raise ValueError(f"channel must be one of {sorted(allowed)}")
        return value

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, value: str) -> str:
        if value not in {"in", "out"}:
            raise ValueError("direction must be 'in' or 'out'")
        return value


class DataQualityError(ValueError):
    """Raised when quality checks fail."""


def validate_schema(df: pd.DataFrame) -> None:
    missing_columns = sorted(set(REQUIRED_TXN_COLUMNS) - set(df.columns))
    if missing_columns:
        raise DataQualityError(f"Missing required columns: {missing_columns}")

    for row_num, record in enumerate(df[REQUIRED_TXN_COLUMNS].to_dict(orient="records"), start=1):
        try:
            TransactionRecord.model_validate(record)
        except ValidationError as exc:
            raise DataQualityError(f"Schema validation failed at row {row_num}: {exc}") from exc


def run_quality_checks(df: pd.DataFrame, now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)

    mandatory_cols = [col for col in REQUIRED_TXN_COLUMNS if col != "beneficiary_account"]
    missing_breakdown = df[mandatory_cols].isna().sum()
    missing_total = int(missing_breakdown.sum())
    if missing_total > 0:
        offenders = missing_breakdown[missing_breakdown > 0].to_dict()
        raise DataQualityError(f"Missing values found: {offenders}")

    transfer_missing_beneficiary = (
        df["channel"].eq("online_transfer") & df["beneficiary_account"].isna()
    )
    if transfer_missing_beneficiary.any():
        raise DataQualityError(
            "Missing beneficiary_account found for online_transfer records: "
            f"{int(transfer_missing_beneficiary.sum())}"
        )

    duplicate_txns = int(df.duplicated(subset=["txn_id"]).sum())
    if duplicate_txns > 0:
        raise DataQualityError(f"Duplicate txn_id values found: {duplicate_txns}")

    if (df["amount"] <= 0).any():
        bad_count = int((df["amount"] <= 0).sum())
        raise DataQualityError(f"Non-positive amounts found: {bad_count}")

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if ts.isna().any():
        bad_count = int(ts.isna().sum())
        raise DataQualityError(f"Unparseable timestamps found: {bad_count}")

    too_old = ts < pd.Timestamp("2020-01-01", tz="UTC")
    too_future = ts > (pd.Timestamp(now) + pd.Timedelta(days=1))
    if too_old.any() or too_future.any():
        old_count = int(too_old.sum())
        future_count = int(too_future.sum())
        raise DataQualityError(
            "Impossible timestamps found "
            f"(before 2020-01-01: {old_count}, later than now+1day: {future_count})"
        )

    direction_bad = ~df["direction"].isin(["in", "out"])
    if direction_bad.any():
        raise DataQualityError(f"Invalid direction values found: {int(direction_bad.sum())}")


def validate_transactions(df: pd.DataFrame) -> None:
    validate_schema(df)
    run_quality_checks(df)


def build_dq_summary(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    required_non_optional = [col for col in REQUIRED_TXN_COLUMNS if col != "beneficiary_account"]
    missing_required = int(df[required_non_optional].isna().sum().sum())
    checks = [
        {
            "check_name": "missing_required_values",
            "status": "pass" if missing_required == 0 else "fail",
            "value": missing_required,
            "details": "Null count across required non-optional columns.",
        },
        {
            "check_name": "duplicate_txn_id",
            "status": "pass" if int(df.duplicated(subset=["txn_id"]).sum()) == 0 else "fail",
            "value": int(df.duplicated(subset=["txn_id"]).sum()),
            "details": "Duplicate count for txn_id.",
        },
        {
            "check_name": "non_positive_amount",
            "status": "pass" if int((df["amount"] <= 0).sum()) == 0 else "fail",
            "value": int((df["amount"] <= 0).sum()),
            "details": "Count of amounts <= 0.",
        },
        {
            "check_name": "invalid_timestamp",
            "status": "pass" if int(ts.isna().sum()) == 0 else "fail",
            "value": int(ts.isna().sum()),
            "details": "Count of unparseable timestamps.",
        },
        {
            "check_name": "transfer_without_beneficiary",
            "status": "pass"
            if int((df["channel"].eq("online_transfer") & df["beneficiary_account"].isna()).sum())
            == 0
            else "fail",
            "value": int(
                (df["channel"].eq("online_transfer") & df["beneficiary_account"].isna()).sum()
            ),
            "details": "Count of online transfers with missing beneficiary account.",
        },
    ]
    return pd.DataFrame(checks)
