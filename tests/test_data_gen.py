from __future__ import annotations

from econ_crime_monitoring_lab.data_gen import generate_synthetic_data


def test_generate_synthetic_data_small_sample_returns_non_empty_outputs() -> None:
    bundle = generate_synthetic_data(days=1, n_customers=5, seed=1)

    assert not bundle["accounts"].empty
    assert not bundle["transactions"].empty
