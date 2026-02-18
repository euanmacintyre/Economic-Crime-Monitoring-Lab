from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for local synthetic pipeline runs."""

    model_config = SettingsConfigDict(
        env_prefix="ECON_CRIME_",
        env_file=".env",
        extra="ignore",
    )

    db_path: Path = Field(default=Path("data/econ_crime_lab.db"), alias="DB_PATH")
    reports_dir: Path = Field(default=Path("reports"), alias="REPORTS_DIR")
    seed: int = Field(default=42, alias="SEED")
    days: int = Field(default=90, alias="DAYS")
    n_customers: int = Field(default=2000, alias="N_CUSTOMERS")
    target_alert_rate: float | None = Field(default=None, alias="TARGET_ALERT_RATE")


def get_settings() -> Settings:
    return Settings()

