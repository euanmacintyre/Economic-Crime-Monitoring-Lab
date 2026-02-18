from __future__ import annotations

import pandas as pd

RULES: dict[str, dict[str, object]] = {
    "velocity_burst": {
        "weight": 14,
        "description": "High customer transaction count in the prior hour.",
        "threshold": "velocity_1h >= 5",
    },
    "amount_spike_vs_baseline": {
        "weight": 16,
        "description": "Amount is materially above customer historical baseline.",
        "threshold": "amount_to_customer_mean >= 4.0 and amount >= 400",
    },
    "new_device_high_value": {
        "weight": 14,
        "description": "High-value transaction on a newly seen device.",
        "threshold": "is_new_device == 1 and amount >= 1500",
    },
    "new_country_activity": {
        "weight": 10,
        "description": "Activity from a newly seen country for the customer.",
        "threshold": "is_new_country == 1 and is_unusual_country == 1",
    },
    "small_then_large_pattern": {
        "weight": 15,
        "description": "Many small transactions followed by a larger transaction.",
        "threshold": "small_txn_24h >= 3 and amount >= 250",
    },
    "rapid_in_out_transfer": {
        "weight": 18,
        "description": "Outgoing transfer shortly after similar incoming transfer.",
        "threshold": "rapid_in_then_out == 1",
    },
    "cash_like_cluster": {
        "weight": 8,
        "description": "Repeated high-value activity with cash-like merchants.",
        "threshold": "merchant_cash_like == 1 and amount >= 200",
    },
    "cross_border_transfer": {
        "weight": 12,
        "description": "Cross-border outgoing transfer above threshold.",
        "threshold": "channel == online_transfer and direction == out and amount >= 900",
    },
}


def score_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    conditions: dict[str, pd.Series] = {
        "velocity_burst": out["velocity_1h"] >= 5,
        "amount_spike_vs_baseline": (
            (out["amount_to_customer_mean"] >= 4.0) & (out["amount"] >= 400)
        ),
        "new_device_high_value": (out["is_new_device"] == 1) & (out["amount"] >= 1500),
        "new_country_activity": (out["is_new_country"] == 1) & (out["is_unusual_country"] == 1),
        "small_then_large_pattern": (out["small_txn_24h"] >= 3) & (out["amount"] >= 250),
        "rapid_in_out_transfer": out["rapid_in_then_out"] == 1,
        "cash_like_cluster": out["merchant_cash_like"].eq(1)
        & out["channel"].isin(["card_present", "atm_cash"])
        & (out["amount"] >= 200),
        "cross_border_transfer": out["channel"].eq("online_transfer")
        & out["direction"].eq("out")
        & out["is_unusual_country"].eq(1)
        & (out["amount"] >= 900),
    }

    risk_score = pd.Series(0, index=out.index, dtype=float)
    for rule_name, cond in conditions.items():
        out[f"rule_{rule_name}"] = cond.astype(int)
        risk_score += cond.astype(int) * int(RULES[rule_name]["weight"])

    out["rules_risk_score"] = risk_score.clip(upper=100)
    out["rule_hit_count"] = pd.DataFrame(
        {rule: cond.astype(int) for rule, cond in conditions.items()}
    ).sum(axis=1)

    def _hits_for_row(row: pd.Series) -> str:
        hits = [rule for rule in conditions if row[f"rule_{rule}"] == 1]
        return ", ".join(hits) if hits else "none"

    out["rule_hits"] = out.apply(_hits_for_row, axis=1)

    return out


def get_rule_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rule_name": name,
                "weight": config["weight"],
                "description": config["description"],
                "threshold": config["threshold"],
            }
            for name, config in RULES.items()
        ]
    )


def get_rule_catalog() -> dict[str, dict[str, object]]:
    return RULES.copy()

