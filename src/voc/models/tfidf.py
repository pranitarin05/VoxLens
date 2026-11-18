"""PRD 7.3: TF-IDF + Logistic Regression baseline.

TfidfVectorizer (word 1-2 grams, min_df=5, sublinear_tf, capped vocabulary)
then LogisticRegression with class_weight='balanced'. C is tuned on the
validation set only, using macro-F1. One CLI: ``train`` / ``evaluate``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import joblib
import mlflow
import numpy as np
import pandas as pd
import typer
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from voc.evaluation.compare import CSV_PATH, MD_PATH, evaluate_and_log, upsert_comparison_row
from voc.evaluation.cost import load_pricing
from voc.models.registry import configure

app = typer.Typer(help="PRD 7.3 TF-IDF + Logistic Regression baseline.")

PROCESSED = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "tfidf.joblib"
CONFIGS = Path("configs")
SEED = 42
C_GRID = (0.1, 1.0, 10.0)
MAX_FEATURES = 500_000


def load_split(name: str, processed: Path = PROCESSED) -> pd.DataFrame:
    """Load a processed split parquet (needs clean_text + label)."""
    path = processed / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run make splits first")
    frame = pd.read_parquet(path, columns=["clean_text", "label"])
    frame = frame.dropna(subset=["clean_text"])
    return frame


def build_pipeline(c_value: float, seed: int = SEED) -> Pipeline:
    """Fresh TF-IDF + LogisticRegression pipeline for one C value."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=5,
                    sublinear_tf=True,
                    max_features=MAX_FEATURES,
                    dtype=np.float32,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )


def train(
    train_split: str = "train_200k",
    val_split: str = "val_20k",
    model_path: Path = MODEL_PATH,
) -> dict[str, object]:
    """Tune C on validation macro-F1, save the best pipeline, log to MLflow."""
    train_frame = load_split(train_split)
    val_frame = load_split(val_split)
    x_train, y_train = train_frame["clean_text"], train_frame["label"]
    x_val, y_val = val_frame["clean_text"], val_frame["label"]

    configure()
    best_c: float | None = None
    best_f1 = -1.0
    best_pipeline: Pipeline | None = None

    with mlflow.start_run(run_name="tfidf-train") as run:
        mlflow.log_params(
            {
                "model": "tfidf-logreg",
                "seed": SEED,
                "ngram_range": "1-2",
                "min_df": 5,
                "max_features": MAX_FEATURES,
                "train_split": train_split,
                "val_split": val_split,
                "n_train": len(train_frame),
            }
        )
        for c_value in C_GRID:
            pipeline = build_pipeline(c_value)
            pipeline.fit(x_train, y_train)
            val_pred = pipeline.predict(x_val)
            from sklearn.metrics import f1_score

            macro_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
            mlflow.log_metric(f"val_macro_f1_C{c_value}", macro_f1)
            typer.echo(f"C={c_value}: val macro-F1={macro_f1:.4f}")
            if macro_f1 > best_f1:
                best_f1, best_c, best_pipeline = macro_f1, c_value, pipeline

        assert best_pipeline is not None and best_c is not None
        mlflow.log_params({"best_C": best_c})
        mlflow.log_metric("val_macro_f1", best_f1)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model_file")
        try:
            mlflow.sklearn.log_model(best_pipeline, name="model")
        except Exception as exc:  # noqa: BLE001 - logging path differs across versions
            typer.echo(f"warning: sklearn.log_model failed: {exc}")

    typer.echo(f"best C={best_c} val macro-F1={best_f1:.4f} -> {model_path}")
    return {
        "best_c": best_c,
        "val_macro_f1": best_f1,
        "run_id": run.info.run_id,
        "model_path": str(model_path),
    }


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    """Load the saved TF-IDF pipeline."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `make eval-tfidf` first")
    return joblib.load(path)


def make_predict_fn(model: Pipeline):
    """Return a batch predict function (list[str] -> list[str])."""

    def predict_fn(texts: Sequence[str]) -> list[str]:
        return list(model.predict(list(texts)))

    return predict_fn


def predict_proba(text: str, model: Pipeline | None = None) -> dict[str, object]:
    """Predict label + class probabilities for one review."""
    model = model or load_model()
    proba = model.predict_proba([text])[0]
    classes = list(model.classes_)
    best = int(np.argmax(proba))
    return {
        "label": classes[best],
        "probabilities": {cls: float(p) for cls, p in zip(classes, proba, strict=False)},
    }


@app.command("train")
def train_cmd(
    train_split: Annotated[str, typer.Option()] = "train_200k",
    val_split: Annotated[str, typer.Option()] = "val_20k",
) -> None:
    """Tune C and save the most accurate TF-IDF baseline."""
    train(train_split=train_split, val_split=val_split)


@app.command()
def evaluate(
    datasets: Annotated[str, typer.Option(help="Comma-separated split names")] = (
        "llm_test_1500,test_50k"
    ),
) -> None:
    """Score the saved model on eval splits and update the comparison files."""
    model = load_model()
    predict_fn = make_predict_fn(model)
    pricing = load_pricing(CONFIGS / "pricing.yaml")
    usd_per_hour = pricing.get("hourly_rate_usd")
    version = "tfidf-logreg-v1"

    with open(CONFIGS / "training.yaml", encoding="utf-8") as handle:
        training = yaml.safe_load(handle) or {}

    rows = []
    for dataset in [d.strip() for d in datasets.split(",") if d.strip()]:
        frame = load_split(dataset)
        row = evaluate_and_log(
            predict_fn,
            frame["clean_text"].tolist(),
            frame["label"].tolist(),
            model_name="tfidf",
            model_version=version,
            dataset=dataset,
            usd_per_hour=usd_per_hour,
            extra_params={"seed": SEED, "seed_config": training.get("seed")},
        )
        upsert_comparison_row(row)
        rows.append(row)
        typer.echo(
            f"{dataset}: acc={row['accuracy']:.4f} macroF1={row['macro_f1']:.4f} "
            f"p95={row['single_p95_ms']:.2f}ms"
        )
    typer.echo(f"updated {CSV_PATH} and {MD_PATH}")


@app.command()
def predict(text: Annotated[str, typer.Argument(help="Review text")]) -> None:
    """Print predicted label and probabilities for one review."""
    typer.echo(predict_proba(text))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
