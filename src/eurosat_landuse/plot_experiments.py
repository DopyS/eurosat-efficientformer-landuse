from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

from .config import load_config
from .data import EUROSAT_CLASSES
from .paths import ensure_directory, project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plot-experiments",
        description="Create report-friendly figures from experiment summary CSV.",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    parser.add_argument("--summary", default=None, help="Path to experiment_summary.csv.")
    parser.add_argument("--eval-json", default=None, help="Optional evaluation JSON for class-level figures.")
    parser.add_argument("--output-prefix", default="experiment", help="Figure filename prefix.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    outputs_cfg = dict(config.get("outputs", {}))
    summary_path = project_path(args.summary or Path(outputs_cfg.get("metrics", "outputs/metrics")) / "experiment_summary.csv")
    figures_dir = ensure_directory(project_path(outputs_cfg.get("figures", "outputs/figures")))
    rows = load_training_rows(summary_path)
    if not rows:
        raise RuntimeError(f"No training rows found in {summary_path}.")

    accuracy_path = figures_dir / f"{args.output_prefix}_accuracy.png"
    loss_path = figures_dir / f"{args.output_prefix}_loss.png"
    plot_bar(
        rows,
        y_key="val_accuracy",
        title="Validation Accuracy Comparison",
        ylabel="Validation Accuracy",
        output_path=accuracy_path,
    )
    plot_bar(
        rows,
        y_key="val_loss",
        title="Validation Loss Comparison",
        ylabel="Validation Loss",
        output_path=loss_path,
    )
    print(f"- accuracy figure saved to: {accuracy_path}")
    print(f"- loss figure saved to: {loss_path}")

    if args.eval_json:
        eval_path = project_path(args.eval_json)
        eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
        confusion_path = figures_dir / f"{args.output_prefix}_confusion_matrix.png"
        per_class_path = figures_dir / f"{args.output_prefix}_per_class_accuracy.png"
        plot_confusion_matrix(eval_payload, confusion_path)
        plot_per_class_accuracy(eval_payload, per_class_path)
        print(f"- confusion matrix saved to: {confusion_path}")
        print(f"- per-class accuracy figure saved to: {per_class_path}")
    return 0


def load_training_rows(summary_path: Path) -> list[dict[str, str]]:
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    training_rows = [row for row in rows if row.get("record_type") == "train"]
    return sorted(training_rows, key=lambda row: row.get("run_name", ""))


def plot_bar(rows: list[dict[str, str]], *, y_key: str, title: str, ylabel: str, output_path: Path) -> None:
    matplotlib = __import__("matplotlib")
    matplotlib.use("Agg")
    pyplot = __import__("matplotlib.pyplot").pyplot

    names = [row["run_name"] for row in rows]
    values = [float(row[y_key]) for row in rows]
    width = max(8, len(names) * 1.2)
    _, axis = pyplot.subplots(figsize=(width, 4.8))
    bars = axis.bar(names, values, color="#3f7fbf")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Experiment")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    pyplot.tight_layout()
    pyplot.savefig(output_path, dpi=160)
    pyplot.close()


def plot_confusion_matrix(eval_payload: dict, output_path: Path) -> None:
    matplotlib = __import__("matplotlib")
    matplotlib.use("Agg")
    pyplot = __import__("matplotlib.pyplot").pyplot
    metrics = eval_payload["metrics"]
    matrix = metrics["confusion_matrix"]
    labels = class_labels(len(matrix))

    _, axis = pyplot.subplots(figsize=(8.5, 7.2))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title(f"Confusion Matrix ({eval_payload.get('run_name', 'evaluation')})")
    axis.set_xlabel("Predicted Class")
    axis.set_ylabel("True Class")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels, fontsize=8)
    pyplot.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    pyplot.tight_layout()
    pyplot.savefig(output_path, dpi=170)
    pyplot.close()


def plot_per_class_accuracy(eval_payload: dict, output_path: Path) -> None:
    matplotlib = __import__("matplotlib")
    matplotlib.use("Agg")
    pyplot = __import__("matplotlib.pyplot").pyplot
    per_class = eval_payload["metrics"]["per_class"]
    labels = [row["class_name"] for row in per_class]
    values = [row["accuracy"] if row["accuracy"] is not None else 0.0 for row in per_class]

    _, axis = pyplot.subplots(figsize=(10, 4.8))
    bars = axis.bar(labels, values, color="#4f9f6f")
    axis.set_title(f"Per-class Accuracy ({eval_payload.get('run_name', 'evaluation')})")
    axis.set_ylabel("Accuracy")
    axis.set_xlabel("Class")
    axis.set_ylim(0, 1.05)
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    for bar, value in zip(bars, values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    pyplot.tight_layout()
    pyplot.savefig(output_path, dpi=170)
    pyplot.close()


def class_labels(count: int) -> list[str]:
    return [EUROSAT_CLASSES[index] if index < len(EUROSAT_CLASSES) else str(index) for index in range(count)]


if __name__ == "__main__":
    raise SystemExit(main())
