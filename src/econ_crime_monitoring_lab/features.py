from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd


def build_transaction_features(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    merchants: pd.DataFrame,
    devices: pd.DataFrame,
) -> pd.DataFrame:
    df = transactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    home_country_map = customers.set_index("customer_id")["home_country"].to_dict()
    merchant_cluster_map = merchants.set_index("merchant_id")["merchant_cluster"].to_dict()

    start_ts = df["timestamp"].min()
    baseline_devices = devices.copy()
    baseline_devices["first_seen"] = pd.to_datetime(baseline_devices["first_seen"], utc=True)
    baseline_devices = baseline_devices[baseline_devices["first_seen"] < start_ts]

    seen_devices: dict[str, set[str]] = defaultdict(set)
    for _, row in baseline_devices.iterrows():
        seen_devices[str(row["customer_id"])].add(str(row["device_id"]))

    seen_countries: dict[str, set[str]] = defaultdict(set)
    for customer_id, home_country in home_country_map.items():
        seen_countries[str(customer_id)].add(str(home_country))

    recent_customer_txn_1h: dict[str, deque[pd.Timestamp]] = defaultdict(deque)
    recent_customer_small_24h: dict[str, deque[pd.Timestamp]] = defaultdict(deque)
    recent_account_incoming_60m: dict[str, deque[tuple[pd.Timestamp, float]]] = defaultdict(deque)

    customer_sum_amount: dict[str, float] = defaultdict(float)
    customer_count_amount: dict[str, int] = defaultdict(int)
    customer_last_ts: dict[str, pd.Timestamp] = {}

    velocity_1h: list[int] = []
    small_txn_24h: list[int] = []
    customer_prev_mean_amount: list[float] = []
    amount_to_customer_mean: list[float] = []
    is_new_device: list[int] = []
    is_new_country: list[int] = []
    is_unusual_country: list[int] = []
    rapid_in_then_out: list[int] = []
    merchant_cash_like: list[int] = []
    time_since_prev_customer_min: list[float] = []

    for _, row in df.iterrows():
        customer_id = str(row["customer_id"])
        account_id = str(row["account_id"])
        timestamp = row["timestamp"]
        amount = float(row["amount"])
        country = str(row["country"])
        device_id = str(row["device_id"])
        direction = str(row["direction"])
        channel = str(row["channel"])

        dq_txn = recent_customer_txn_1h[customer_id]
        while dq_txn and (timestamp - dq_txn[0]) > pd.Timedelta(hours=1):
            dq_txn.popleft()

        dq_small = recent_customer_small_24h[customer_id]
        while dq_small and (timestamp - dq_small[0]) > pd.Timedelta(hours=24):
            dq_small.popleft()

        dq_in = recent_account_incoming_60m[account_id]
        while dq_in and (timestamp - dq_in[0][0]) > pd.Timedelta(minutes=60):
            dq_in.popleft()

        prev_count = customer_count_amount[customer_id]
        prev_sum = customer_sum_amount[customer_id]
        prev_mean = prev_sum / prev_count if prev_count else max(amount, 1.0)

        velocity_1h.append(len(dq_txn))
        small_txn_24h.append(len(dq_small))
        customer_prev_mean_amount.append(prev_mean)
        amount_to_customer_mean.append(amount / max(prev_mean, 1.0))

        new_device_flag = int(device_id not in seen_devices[customer_id])
        new_country_flag = int(country not in seen_countries[customer_id])
        unusual_country_flag = int(country != str(home_country_map.get(customer_id, "US")))

        is_new_device.append(new_device_flag)
        is_new_country.append(new_country_flag)
        is_unusual_country.append(unusual_country_flag)

        rapid_flag = 0
        if channel == "online_transfer" and direction == "out":
            for in_ts, in_amount in dq_in:
                ratio = amount / max(in_amount, 1e-6)
                if 0.8 <= ratio <= 1.2:
                    rapid_flag = 1
                    break
        rapid_in_then_out.append(rapid_flag)

        cluster = str(merchant_cluster_map.get(str(row["merchant_id"]), "other"))
        merchant_cash_like.append(int(cluster == "cash_like"))

        last_ts = customer_last_ts.get(customer_id)
        if last_ts is None:
            time_since_prev_customer_min.append(-1.0)
        else:
            delta_min = (timestamp - last_ts).total_seconds() / 60.0
            time_since_prev_customer_min.append(float(delta_min))

        dq_txn.append(timestamp)
        if amount <= 10:
            dq_small.append(timestamp)
        if channel == "online_transfer" and direction == "in":
            dq_in.append((timestamp, amount))

        seen_devices[customer_id].add(device_id)
        seen_countries[customer_id].add(country)
        customer_sum_amount[customer_id] += amount
        customer_count_amount[customer_id] += 1
        customer_last_ts[customer_id] = timestamp

    df["velocity_1h"] = velocity_1h
    df["small_txn_24h"] = small_txn_24h
    df["customer_prev_mean_amount"] = customer_prev_mean_amount
    df["amount_to_customer_mean"] = amount_to_customer_mean
    df["is_new_device"] = is_new_device
    df["is_new_country"] = is_new_country
    df["is_unusual_country"] = is_unusual_country
    df["rapid_in_then_out"] = rapid_in_then_out
    df["merchant_cash_like"] = merchant_cash_like
    df["time_since_prev_customer_min"] = time_since_prev_customer_min

    df["hour"] = df["timestamp"].dt.hour.astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["amount_log"] = np.log1p(df["amount"].astype(float))

    return df

