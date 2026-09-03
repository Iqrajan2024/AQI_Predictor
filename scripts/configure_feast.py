from pathlib import Path
from urllib.parse import quote_plus
import os
import shutil

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env", override=False)

FEAST_REPO = PROJECT_ROOT / "feature_repo" / "feature_repo"

LOCAL_CONFIG = FEAST_REPO / "feature_store.local.yaml"
CLOUD_CONFIG = FEAST_REPO / "feature_store.cloud.yaml"
ACTIVE_CONFIG = FEAST_REPO / "feature_store.yaml"

APP_ENV = os.getenv("APP_ENV", "local").lower()


if APP_ENV in {"cloud", "production", "prod"}:
    if not CLOUD_CONFIG.exists():
        raise FileNotFoundError(
            f"Cloud Feast configuration not found: {CLOUD_CONFIG}"
        )

    postgres_host = os.getenv("FEAST_POSTGRES_HOST")
    postgres_port_raw = os.getenv("FEAST_POSTGRES_PORT", "5432").strip()

    try:
        postgres_port = int(postgres_port_raw)
    except ValueError:
        raise RuntimeError(
            f"FEAST_POSTGRES_PORT must be an integer, got: {postgres_port_raw!r}"
        )
    
    postgres_database = os.getenv(
        "FEAST_POSTGRES_DATABASE",
        "postgres",
    )
    postgres_schema = os.getenv(
        "FEAST_POSTGRES_SCHEMA",
        "feast",
    )
    postgres_user = os.getenv("FEAST_POSTGRES_USER")
    postgres_password = os.getenv("FEAST_POSTGRES_PASSWORD")
    postgres_sslmode = os.getenv(
        "FEAST_POSTGRES_SSLMODE",
        "require",
    )

    required = {
        "FEAST_POSTGRES_HOST": postgres_host,
        "FEAST_POSTGRES_USER": postgres_user,
        "FEAST_POSTGRES_PASSWORD": postgres_password,
    }

    missing = [
        name
        for name, value in required.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            "Missing required Feast cloud environment variables: "
            + ", ".join(missing)
        )

    feast_registry_url = (
        "postgresql+psycopg://"
        f"{quote_plus(postgres_user)}:"
        f"{quote_plus(postgres_password)}@"
        f"{postgres_host}:"
        f"{postgres_port}/"
        f"{postgres_database}"
        "?sslmode=require"
        "&options=-csearch_path%3Dfeast"
    )

    cloud_yaml = f"""project: pearls_aqi

registry:
  registry_type: sql
  path: {feast_registry_url}

provider: local

online_store:
  type: postgres
  host: {postgres_host}
  port: {postgres_port}
  database: {postgres_database}
  db_schema: {postgres_schema}
  user: {postgres_user}
  password: {postgres_password}
  sslmode: {postgres_sslmode}

offline_store:
  type: duckdb
"""

    ACTIVE_CONFIG.write_text(
        cloud_yaml,
        encoding="utf-8",
    )

    print("Configured Feast for cloud environment.")
    print(f"Source: {CLOUD_CONFIG}")
    print(f"Active: {ACTIVE_CONFIG}")

else:
    if not LOCAL_CONFIG.exists():
        raise FileNotFoundError(
            f"Local Feast configuration not found: {LOCAL_CONFIG}"
        )

    shutil.copy2(
        LOCAL_CONFIG,
        ACTIVE_CONFIG,
    )

    print("Configured Feast for local environment.")
    print(f"Source: {LOCAL_CONFIG}")
    print(f"Active: {ACTIVE_CONFIG}")