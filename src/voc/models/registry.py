"""MLflow tracking + model registry helpers (FR-7).

Default tracking store is a local SQLite database so both tracking and the
Model Registry work without an extra service on Windows. Override with
MLFLOW_TRACKING_URI / MLFLOW_EXPERIMENT_NAME.
"""

from __future__ import annotations

import os

import mlflow
from mlflow.tracking import MlflowClient

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "voc-sentiment"


def configure() -> str:
    """Point MLflow at the configured tracking store and experiment; return URI."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT))
    return uri


def register_model(
    run_id: str,
    model_name: str,
    *,
    artifact_path: str = "model",
    alias: str = "production",
) -> str:
    """Register a logged model and assign the alias; return the new version."""
    model_uri = f"runs:/{run_id}/{artifact_path}"
    version = mlflow.register_model(model_uri, model_name)
    client = MlflowClient()
    client.set_registered_model_alias(model_name, alias, version.version)
    return str(version.version)


def get_production_version(model_name: str, alias: str = "production") -> str | None:
    """Return the model version behind an alias, or None if unset."""
    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(model_name, alias)
    except Exception:  # noqa: BLE001 - alias missing raises a registry error
        return None
    return str(version.version)
