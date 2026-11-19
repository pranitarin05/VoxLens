"""PRD 7.4 / FR-5: fine-tune DistilBERT and BERT-base for 3-class sentiment.

Hyperparameters follow configs/training.yaml (max_len 128, lr 2e-5, 2 epochs,
6% warmup, weight decay 0.01, class-weighted cross-entropy, mixed precision on
GPU). Keeps the checkpoint with the best validation macro-F1. Designed for a
4 GB GPU: use gradient accumulation to reach an effective batch of 32.

CLI: ``train`` (fine-tune) and ``evaluate`` (score on eval splits via the
shared FR-6 harness).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import mlflow
import numpy as np
import pandas as pd
import torch
import typer
import yaml
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from voc.evaluation.compare import evaluate_and_log, upsert_comparison_row
from voc.evaluation.cost import load_pricing
from voc.evaluation.metrics import LABELS, compute_metrics
from voc.models.registry import configure, register_model

app = typer.Typer(help="PRD 7.4 transformer fine-tuning (DistilBERT / BERT).")

PROCESSED = Path("data/processed")
MODEL_DIR = Path("models")
CONFIGS = Path("configs")
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
CHUNK_SAVE_STEPS = 50


def load_split(name: str, processed: Path = PROCESSED) -> pd.DataFrame:
    """Load a processed split parquet (clean_text + label)."""
    path = processed / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run make splits first")
    frame = pd.read_parquet(path, columns=["clean_text", "label"])
    frame = frame.dropna(subset=["clean_text"])
    frame = frame[frame["label"].isin(LABEL2ID)]
    return frame


class ReviewDataset(Dataset):
    """Tokenise review text on the fly; labels mapped to fixed ids."""

    def __init__(
        self, texts: Sequence[str], labels: Sequence[str], tokenizer, max_len: int
    ) -> None:
        self.texts = list(texts)
        self.labels = [LABEL2ID[label] for label in labels]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(self.texts[index], truncation=True, max_length=self.max_len)
        encoded["labels"] = self.labels[index]
        return encoded


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy (PRD 4.7 / 7.4)."""

    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # noqa: ANN001
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(outputs.logits.device)
        loss = nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def class_weights_from_labels(labels: Sequence[str]) -> torch.Tensor:
    """Inverse-frequency class weights (balanced), ordered by LABEL2ID."""
    counts = np.array([sum(1 for label in labels if label == name) for name in LABELS], dtype=float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _compute_metrics(eval_pred) -> dict[str, float]:  # noqa: ANN001
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    y_true = [ID2LABEL[int(i)] for i in labels]
    y_pred = [ID2LABEL[int(i)] for i in preds]
    result = compute_metrics(y_true, y_pred)
    return {"accuracy": result["accuracy"], "macro_f1": result["macro_f1"]}


def _load_training_config(path: Path = CONFIGS / "training.yaml") -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def train(
    model_name: str = "distilbert-base-uncased",
    *,
    train_split: str = "train_200k",
    val_split: str = "val_20k",
    sample: int | None = None,
    val_sample: int | None = None,
    epochs: int | None = None,
    batch_size: int = 8,
    grad_accum: int = 4,
    max_len: int | None = None,
    fp16: bool = True,
    output_dir: Path | None = None,
    max_steps: int | None = None,
    resume: bool = False,
    register: bool = False,
) -> dict[str, object]:
    """Fine-tune one transformer; return metrics and the saved model directory."""
    cfg = _load_training_config()
    seed = int(cfg.get("seed", 42))
    max_len = max_len or int(cfg.get("max_len", 128))
    epochs = epochs or int(cfg.get("epochs", 2))
    lr = float(cfg.get("learning_rate", 2e-5))
    warmup = float(cfg.get("warmup_ratio", 0.06))
    weight_decay = float(cfg.get("weight_decay", 0.01))

    train_frame = load_split(train_split)
    val_frame = load_split(val_split)
    if sample:
        train_frame = train_frame.sample(n=min(sample, len(train_frame)), random_state=seed)
    if val_sample:
        val_frame = val_frame.sample(n=min(val_sample, len(val_frame)), random_state=seed)
    train_frame = train_frame.reset_index(drop=True)
    val_frame = val_frame.reset_index(drop=True)

    use_fp16 = bool(fp16 and torch.cuda.is_available())
    device_label = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    typer.echo(
        f"model={model_name} device={device_label} fp16={use_fp16} "
        f"train={len(train_frame)} val={len(val_frame)} epochs={epochs} "
        f"batch={batch_size} grad_accum={grad_accum} max_len={max_len}"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = ReviewDataset(train_frame["clean_text"], train_frame["label"], tokenizer, max_len)
    val_ds = ReviewDataset(val_frame["clean_text"], val_frame["label"], tokenizer, max_len)
    weights = class_weights_from_labels(train_frame["label"].tolist())

    out = output_dir or (MODEL_DIR / model_name.split("/")[-1])
    out.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = max(1, -(-len(train_frame) // (batch_size * grad_accum)))
    total_steps = max_steps or (steps_per_epoch * epochs)
    warmup_steps = int(round(warmup * total_steps))

    chunk_mode = bool(max_steps)
    args = TrainingArguments(
        output_dir=str(out),
        learning_rate=lr,
        num_train_epochs=float(epochs),
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(batch_size, 16),
        gradient_accumulation_steps=grad_accum,
        max_steps=max_steps or -1,
        eval_strategy="steps" if chunk_mode else "epoch",
        eval_steps=CHUNK_SAVE_STEPS if chunk_mode else None,
        save_strategy="steps" if chunk_mode else "epoch",
        save_steps=CHUNK_SAVE_STEPS if chunk_mode else 500,
        logging_strategy="steps",
        logging_steps=25,
        load_best_model_at_end=not chunk_mode,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=seed,
        data_seed=seed,
        fp16=use_fp16,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = WeightedTrainer(
        class_weights=weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
    )

    configure()
    with mlflow.start_run(run_name=f"train-{model_name.split('/')[-1]}") as run:
        mlflow.log_params(
            {
                "model": model_name,
                "seed": seed,
                "max_len": max_len,
                "epochs": epochs,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "effective_batch": batch_size * grad_accum,
                "learning_rate": lr,
                "usage_fp16": use_fp16,
                "device": device_label,
                "n_train": len(train_frame),
                "n_val": len(val_frame),
                "max_steps": max_steps or 0,
                "resumed": resume,
            }
        )
        train_result = trainer.train(resume_from_checkpoint=True if resume else None)
        eval_metrics = trainer.evaluate()
        mlflow.log_metrics(
            {k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))}
        )
        mlflow.log_metric("train_runtime_s", float(train_result.metrics.get("train_runtime", 0.0)))
        mlflow.log_metric(
            "train_samples_per_s", float(train_result.metrics.get("train_samples_per_second", 0.0))
        )

        trainer.save_model(str(out))
        tokenizer.save_pretrained(str(out))
        mlflow.log_artifact(str(out), artifact_path="model_dir")
        run_id = run.info.run_id

    version = ""
    if register:
        version = register_model(run_id, model_name.split("/")[-1])

    val_f1 = float(eval_metrics.get("eval_macro_f1", 0.0))
    typer.echo(
        f"saved {out} | val macro-F1={val_f1:.4f} | runtime_s="
        f"{float(train_result.metrics.get('train_runtime', 0.0)):.1f}"
        + (f" | registered version={version}" if version else "")
    )
    return {
        "model_name": model_name,
        "model_dir": str(out),
        "val_macro_f1": val_f1,
        "run_id": run_id,
        "registered_version": version,
    }


def load_model(model_dir: Path):
    """Load a saved model+tokenizer from a directory."""
    if not model_dir.exists():
        raise FileNotFoundError(f"{model_dir} missing — train first")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    return model, tokenizer


def make_predict_fn(model, tokenizer, max_len: int = 128, batch_size: int = 32):
    """Return a batch predict function (list[str] -> list[str]) on GPU if available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def predict_fn(texts: Sequence[str]) -> list[str]:
        outputs: list[str] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            enc = tokenizer(
                batch,
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                logits = model(**enc).logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            outputs.extend(ID2LABEL[int(p)] for p in preds)
        return outputs

    return predict_fn


@app.command("train")
def train_cmd(
    model_name: Annotated[str, typer.Option()] = "distilbert-base-uncased",
    train_split: Annotated[str, typer.Option()] = "train_200k",
    val_split: Annotated[str, typer.Option()] = "val_20k",
    sample: Annotated[int, typer.Option(help="cap training rows (0 = all)")] = 0,
    val_sample: Annotated[int, typer.Option(help="cap val rows (0 = all)")] = 0,
    epochs: Annotated[int, typer.Option()] = 0,
    batch_size: Annotated[int, typer.Option()] = 8,
    grad_accum: Annotated[int, typer.Option()] = 4,
    output_dir: Annotated[str, typer.Option(help="override save dir")] = "",
    max_steps: Annotated[int, typer.Option(help="absolute optimizer steps (0 = full epochs)")] = 0,
    resume: Annotated[bool, typer.Option(help="resume from latest checkpoint")] = False,
    register: Annotated[bool, typer.Option()] = False,
) -> None:
    """Fine-tune DistilBERT/BERT with the PRD 7.4 recipe."""
    train(
        model_name,
        train_split=train_split,
        val_split=val_split,
        sample=sample or None,
        val_sample=val_sample or None,
        epochs=epochs or None,
        batch_size=batch_size,
        grad_accum=grad_accum,
        output_dir=Path(output_dir) if output_dir else None,
        max_steps=max_steps or None,
        resume=resume,
        register=register,
    )


@app.command("evaluate")
def evaluate_cmd(
    model_dir: Annotated[str, typer.Option()] = "models/distilbert-base-uncased",
    datasets: Annotated[str, typer.Option()] = "llm_test_1500,test_50k",
) -> None:
    """Score a saved transformer and update results/model_comparison.*."""
    model, tokenizer = load_model(Path(model_dir))
    predict_fn = make_predict_fn(model, tokenizer)
    pricing = load_pricing(CONFIGS / "pricing.yaml")
    usd_per_hour = pricing.get("hourly_rate_usd")
    model_label = Path(model_dir).name

    for dataset in [d.strip() for d in datasets.split(",") if d.strip()]:
        frame = load_split(dataset)
        row = evaluate_and_log(
            predict_fn,
            frame["clean_text"].tolist(),
            frame["label"].tolist(),
            model_name=model_label,
            model_version=model_label,
            dataset=dataset,
            usd_per_hour=usd_per_hour,
        )
        upsert_comparison_row(row)
        typer.echo(
            f"{dataset}: acc={row['accuracy']:.4f} macroF1={row['macro_f1']:.4f} "
            f"p95={row['single_p95_ms']:.2f}ms"
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
