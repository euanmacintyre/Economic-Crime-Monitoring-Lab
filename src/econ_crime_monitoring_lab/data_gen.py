from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COUNTRIES = ["US", "GB", "DE", "FR", "ES", "NL", "AE", "SG", "IE", "CA"]
CURRENCIES = ["USD", "EUR", "GBP"]
CHANNELS = ["card_present", "card_not_present", "online_transfer", "atm_cash"]


def _id_series(prefix: str, size: int, width: int = 6) -> list[str]:
    return [f"{prefix}{i:0{width}d}" for i in range(1, size + 1)]


def _build_customers(
    n_customers: int,
    start_ts: pd.Timestamp,
    rng: np.random.Generator,
) -> pd.DataFrame:
    customer_ids = _id_series("C", n_customers)
    created_days_ago = rng.integers(365, 365 * 5, size=n_customers)
    created_at = start_ts - pd.to_timedelta(created_days_ago, unit="D")
    risk_band = rng.choice(["low", "medium", "high"], size=n_customers, p=[0.7, 0.25, 0.05])
    home_country = rng.choice(
        COUNTRIES,
        size=n_customers,
        p=[0.55, 0.05, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02, 0.03, 0.12],
    )
    return pd.DataFrame(
        {
            "customer_id": customer_ids,
            "created_at": created_at,
            "risk_band": risk_band,
            "home_country": home_country,
        }
    )


def _build_accounts(
    customers: pd.DataFrame,
    start_ts: pd.Timestamp,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    account_counter = 1
    for customer_id in customers["customer_id"]:
        n_accounts = int(rng.choice([1, 2, 3], p=[0.72, 0.22, 0.06]))
        for _ in range(n_accounts):
            rows.append(
                {
                    "account_id": f"A{account_counter:07d}",
                    "customer_id": customer_id,
                    "account_type": rng.choice(
                        ["checking", "savings", "credit"], p=[0.55, 0.25, 0.20]
                    ),
                    "opened_at": start_ts - pd.Timedelta(days=int(rng.integers(60, 365 * 6))),
                    "balance": float(np.clip(rng.normal(4500, 2200), 200, 45000)),
                }
            )
            account_counter += 1
    return pd.DataFrame(rows)


def _build_merchants(rng: np.random.Generator) -> pd.DataFrame:
    merchant_ids = _id_series("M", 180)
    merchant_clusters = rng.choice(
        ["retail", "ecommerce", "travel", "utilities", "cash_like", "grocery", "restaurant"],
        size=180,
        p=[0.22, 0.21, 0.08, 0.14, 0.07, 0.15, 0.13],
    )
    merchant_country = rng.choice(
        COUNTRIES,
        size=180,
        p=[0.62, 0.06, 0.05, 0.04, 0.03, 0.03, 0.04, 0.03, 0.03, 0.07],
    )
    merchants = pd.DataFrame(
        {
            "merchant_id": merchant_ids,
            "merchant_name": [f"Merchant {i:03d}" for i in range(1, 181)],
            "merchant_cluster": merchant_clusters,
            "country": merchant_country,
        }
    )

    transfer_row = pd.DataFrame(
        [
            {
                "merchant_id": "M_TRANSFER",
                "merchant_name": "Transfer Network",
                "merchant_cluster": "transfer",
                "country": "US",
            }
        ]
    )
    atm_row = pd.DataFrame(
        [
            {
                "merchant_id": "M_ATM",
                "merchant_name": "ATM Network",
                "merchant_cluster": "cash_like",
                "country": "US",
            }
        ]
    )
    return pd.concat([merchants, transfer_row, atm_row], ignore_index=True)


def _build_devices(
    customers: pd.DataFrame,
    start_ts: pd.Timestamp,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    device_counter = 1
    for customer_id in customers["customer_id"]:
        n_devices = int(rng.choice([1, 2, 3], p=[0.62, 0.30, 0.08]))
        for _ in range(n_devices):
            rows.append(
                {
                    "device_id": f"D{device_counter:08d}",
                    "customer_id": customer_id,
                    "first_seen": start_ts - pd.Timedelta(days=int(rng.integers(10, 365 * 2))),
                    "device_type": rng.choice(
                        ["mobile", "desktop", "tablet"], p=[0.66, 0.28, 0.06]
                    ),
                }
            )
            device_counter += 1
    return pd.DataFrame(rows)


def _pick_device(
    customer_id: str,
    customer_devices: dict[str, list[str]],
    rng: np.random.Generator,
) -> str:
    pool = customer_devices.get(customer_id, [])
    if not pool:
        return "D_UNKNOWN"
    return str(rng.choice(pool))


def _sample_amount(channel: str, rng: np.random.Generator) -> float:
    if channel == "online_transfer":
        return float(np.clip(rng.lognormal(mean=5.7, sigma=0.95), 25, 12000))
    if channel == "atm_cash":
        return float(np.clip(rng.gamma(shape=2.2, scale=55), 10, 1800))
    if channel == "card_not_present":
        return float(np.clip(rng.gamma(shape=2.0, scale=28), 1, 1200))
    return float(np.clip(rng.gamma(shape=2.2, scale=35), 1, 1800))


def _generate_base_transactions(
    customers: pd.DataFrame,
    accounts: pd.DataFrame,
    merchants: pd.DataFrame,
    devices: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    txns_per_day: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, int]:
    n_txns = txns_per_day * max(1, int((end_ts - start_ts).days))

    customer_ids = customers["customer_id"].to_numpy()
    account_map = accounts.groupby("customer_id")["account_id"].apply(list).to_dict()
    customer_home = customers.set_index("customer_id")["home_country"].to_dict()
    customer_devices = devices.groupby("customer_id")["device_id"].apply(list).to_dict()

    card_merchants = merchants[
        merchants["merchant_cluster"].isin(
            ["retail", "ecommerce", "travel", "grocery", "restaurant", "cash_like"]
        )
    ]["merchant_id"].tolist()

    ts_seconds = rng.integers(0, int((end_ts - start_ts).total_seconds()), size=n_txns)
    timestamps = start_ts + pd.to_timedelta(ts_seconds, unit="s")

    rows: list[dict[str, object]] = []
    for i in range(n_txns):
        customer_id = str(rng.choice(customer_ids))
        account_id = str(rng.choice(account_map[customer_id]))
        channel = str(rng.choice(CHANNELS, p=[0.36, 0.28, 0.24, 0.12]))
        direction = "out"
        beneficiary_account: str | None = None
        if channel == "online_transfer":
            direction = str(rng.choice(["out", "in"], p=[0.72, 0.28]))
            if direction == "out":
                beneficiary_account = str(rng.choice(accounts["account_id"].to_numpy()))
                while beneficiary_account == account_id:
                    beneficiary_account = str(rng.choice(accounts["account_id"].to_numpy()))
            else:
                beneficiary_account = str(rng.choice(accounts["account_id"].to_numpy()))
            merchant_id = "M_TRANSFER"
        elif channel == "atm_cash":
            merchant_id = "M_ATM"
        else:
            merchant_id = str(rng.choice(card_merchants))

        home_country = str(customer_home[customer_id])
        country = home_country if rng.random() < 0.9 else str(rng.choice(COUNTRIES))

        rows.append(
            {
                "txn_id": f"T{i + 1:010d}",
                "timestamp": timestamps[i],
                "amount": round(_sample_amount(channel, rng), 2),
                "currency": str(rng.choice(CURRENCIES, p=[0.72, 0.17, 0.11])),
                "customer_id": customer_id,
                "account_id": account_id,
                "merchant_id": merchant_id,
                "channel": channel,
                "country": country,
                "device_id": _pick_device(customer_id, customer_devices, rng),
                "beneficiary_account": beneficiary_account,
                "direction": direction,
                "typology_label": "none",
                "is_suspicious": 0,
            }
        )

    return pd.DataFrame(rows), n_txns


def _append_typology_txn(
    rows: list[dict[str, object]],
    txn_counter: int,
    timestamp: pd.Timestamp,
    amount: float,
    currency: str,
    customer_id: str,
    account_id: str,
    merchant_id: str,
    channel: str,
    country: str,
    device_id: str,
    beneficiary_account: str | None,
    direction: str,
    typology: str,
) -> int:
    rows.append(
        {
            "txn_id": f"T{txn_counter:010d}",
            "timestamp": timestamp,
            "amount": round(float(amount), 2),
            "currency": currency,
            "customer_id": customer_id,
            "account_id": account_id,
            "merchant_id": merchant_id,
            "channel": channel,
            "country": country,
            "device_id": device_id,
            "beneficiary_account": beneficiary_account,
            "direction": direction,
            "typology_label": typology,
            "is_suspicious": 1,
        }
    )
    return txn_counter + 1


def _inject_typologies(
    base_txns: pd.DataFrame,
    customers: pd.DataFrame,
    accounts: pd.DataFrame,
    merchants: pd.DataFrame,
    devices: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    next_txn_number: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    account_map = accounts.groupby("customer_id")["account_id"].apply(list).to_dict()
    home_country = customers.set_index("customer_id")["home_country"].to_dict()
    customer_devices = devices.groupby("customer_id")["device_id"].apply(list).to_dict()
    all_accounts = accounts["account_id"].tolist()

    cash_like_merchants = merchants[merchants["merchant_cluster"] == "cash_like"][
        "merchant_id"
    ].tolist()
    cnp_merchants = merchants[
        merchants["merchant_cluster"].isin(["ecommerce", "retail"])
    ]["merchant_id"].tolist()

    rows: list[dict[str, object]] = []
    new_devices_rows: list[dict[str, object]] = []
    txn_counter = next_txn_number + 1

    customer_ids = customers["customer_id"].tolist()

    n_bursts = max(6, int((end_ts - start_ts).days // 12))
    for _ in range(n_bursts):
        customer_id = str(rng.choice(customer_ids))
        account_id = str(rng.choice(account_map[customer_id]))
        merchant_id = str(rng.choice(cnp_merchants))
        device_id = _pick_device(customer_id, customer_devices, rng)
        base_time = start_ts + pd.Timedelta(
            minutes=int(rng.integers(0, int((end_ts - start_ts).total_seconds() // 60 - 90)))
        )
        small_count = int(rng.integers(4, 8))
        for i in range(small_count):
            txn_counter = _append_typology_txn(
                rows,
                txn_counter,
                base_time + pd.Timedelta(minutes=i * int(rng.integers(1, 3))),
                amount=rng.uniform(1.0, 6.0),
                currency="USD",
                customer_id=customer_id,
                account_id=account_id,
                merchant_id=merchant_id,
                channel="card_not_present",
                country=str(home_country[customer_id]),
                device_id=device_id,
                beneficiary_account=None,
                direction="out",
                typology="card_testing_burst",
            )
        txn_counter = _append_typology_txn(
            rows,
            txn_counter,
            base_time + pd.Timedelta(minutes=small_count * 2),
            amount=rng.uniform(120, 340),
            currency="USD",
            customer_id=customer_id,
            account_id=account_id,
            merchant_id=merchant_id,
            channel="card_not_present",
            country=str(home_country[customer_id]),
            device_id=device_id,
            beneficiary_account=None,
            direction="out",
            typology="card_testing_burst",
        )

    n_ato = max(12, int((end_ts - start_ts).days // 7))
    for i in range(n_ato):
        customer_id = str(rng.choice(customer_ids))
        account_id = str(rng.choice(account_map[customer_id]))
        foreign_country = str(rng.choice([c for c in COUNTRIES if c != home_country[customer_id]]))
        txn_time = start_ts + pd.Timedelta(
            minutes=int(rng.integers(0, int((end_ts - start_ts).total_seconds() // 60)))
        )
        new_device = f"D_NEW_{i + 1:06d}"
        new_devices_rows.append(
            {
                "device_id": new_device,
                "customer_id": customer_id,
                "first_seen": txn_time,
                "device_type": rng.choice(["mobile", "desktop", "tablet"], p=[0.65, 0.3, 0.05]),
            }
        )
        beneficiary = str(rng.choice(all_accounts))
        while beneficiary == account_id:
            beneficiary = str(rng.choice(all_accounts))

        txn_counter = _append_typology_txn(
            rows,
            txn_counter,
            txn_time,
            amount=rng.uniform(2200, 9500),
            currency="USD",
            customer_id=customer_id,
            account_id=account_id,
            merchant_id="M_TRANSFER",
            channel="online_transfer",
            country=foreign_country,
            device_id=new_device,
            beneficiary_account=beneficiary,
            direction="out",
            typology="account_takeover",
        )

    n_mule = max(10, int((end_ts - start_ts).days // 9))
    for _ in range(n_mule):
        account_row = accounts.sample(n=1, random_state=int(rng.integers(1, 1_000_000))).iloc[0]
        account_id = str(account_row["account_id"])
        customer_id = str(account_row["customer_id"])
        device_id = _pick_device(customer_id, customer_devices, rng)
        txn_time_in = start_ts + pd.Timedelta(
            minutes=int(rng.integers(0, int((end_ts - start_ts).total_seconds() // 60 - 60)))
        )
        incoming_amt = float(rng.uniform(900, 5500))
        outgoing_amt = incoming_amt * float(rng.uniform(0.88, 0.99))

        source_account = str(rng.choice(all_accounts))
        while source_account == account_id:
            source_account = str(rng.choice(all_accounts))

        beneficiary = str(rng.choice(all_accounts))
        while beneficiary == account_id:
            beneficiary = str(rng.choice(all_accounts))

        txn_counter = _append_typology_txn(
            rows,
            txn_counter,
            txn_time_in,
            amount=incoming_amt,
            currency="USD",
            customer_id=customer_id,
            account_id=account_id,
            merchant_id="M_TRANSFER",
            channel="online_transfer",
            country=str(home_country[customer_id]),
            device_id=device_id,
            beneficiary_account=source_account,
            direction="in",
            typology="mule_pass_through",
        )
        txn_counter = _append_typology_txn(
            rows,
            txn_counter,
            txn_time_in + pd.Timedelta(minutes=int(rng.integers(5, 45))),
            amount=outgoing_amt,
            currency="USD",
            customer_id=customer_id,
            account_id=account_id,
            merchant_id="M_TRANSFER",
            channel="online_transfer",
            country=str(home_country[customer_id]),
            device_id=device_id,
            beneficiary_account=beneficiary,
            direction="out",
            typology="mule_pass_through",
        )

    n_cash = max(8, int((end_ts - start_ts).days // 10))
    for _ in range(n_cash):
        customer_id = str(rng.choice(customer_ids))
        account_id = str(rng.choice(account_map[customer_id]))
        device_id = _pick_device(customer_id, customer_devices, rng)
        merchant_id = str(rng.choice(cash_like_merchants))
        start_cluster = start_ts + pd.Timedelta(
            minutes=int(rng.integers(0, int((end_ts - start_ts).total_seconds() // 60 - 90)))
        )
        for step in range(int(rng.integers(3, 6))):
            txn_counter = _append_typology_txn(
                rows,
                txn_counter,
                start_cluster + pd.Timedelta(minutes=step * int(rng.integers(10, 35))),
                amount=rng.uniform(180, 900),
                currency="USD",
                customer_id=customer_id,
                account_id=account_id,
                merchant_id=merchant_id,
                channel="card_present",
                country=str(home_country[customer_id]),
                device_id=device_id,
                beneficiary_account=None,
                direction="out",
                typology="cash_merchant_cluster",
            )

    injected = pd.DataFrame(rows)
    if new_devices_rows:
        devices = pd.concat([devices, pd.DataFrame(new_devices_rows)], ignore_index=True)

    txns = pd.concat([base_txns, injected], ignore_index=True)
    txns["timestamp"] = pd.to_datetime(txns["timestamp"], utc=True)
    txns = txns.sort_values("timestamp").reset_index(drop=True)

    return txns, devices


def generate_synthetic_data(
    days: int = 90,
    seed: int = 42,
    txns_per_day: int = 450,
    n_customers: int = 500,
    end_ts: pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    if end_ts is not None:
        resolved_end_ts = pd.Timestamp(end_ts)
        if resolved_end_ts.tzinfo is None:
            resolved_end_ts = resolved_end_ts.tz_localize("UTC")
        else:
            resolved_end_ts = resolved_end_ts.tz_convert("UTC")
        resolved_end_ts = resolved_end_ts.floor("min")
    else:
        resolved_end_ts = pd.Timestamp.now(tz="UTC").floor("min")
    start_ts = resolved_end_ts - pd.Timedelta(days=days)

    customers = _build_customers(n_customers=n_customers, start_ts=start_ts, rng=rng)
    accounts = _build_accounts(customers=customers, start_ts=start_ts, rng=rng)
    merchants = _build_merchants(rng=rng)
    devices = _build_devices(customers=customers, start_ts=start_ts, rng=rng)

    base_txns, next_txn_number = _generate_base_transactions(
        customers=customers,
        accounts=accounts,
        merchants=merchants,
        devices=devices,
        start_ts=start_ts,
        end_ts=resolved_end_ts,
        txns_per_day=txns_per_day,
        rng=rng,
    )
    transactions, devices = _inject_typologies(
        base_txns=base_txns,
        customers=customers,
        accounts=accounts,
        merchants=merchants,
        devices=devices,
        start_ts=start_ts,
        end_ts=resolved_end_ts,
        next_txn_number=next_txn_number,
        rng=rng,
    )

    return {
        "customers": customers,
        "accounts": accounts,
        "merchants": merchants,
        "devices": devices,
        "transactions": transactions,
    }


def save_dataset_bundle(
    bundle: dict[str, pd.DataFrame],
    output_dir: Path | str = Path("data/raw"),
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for name, df in bundle.items():
        df.to_csv(output_path / f"{name}.csv", index=False)
