from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

from .config import load_config
from .paths import ensure_directory, project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plot-experiments",
        description="Create report-friendly figures from experiment summary CSV.",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    parser.add_argument("--summary", default=None, help="Path to experiment_summary.csv.")
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


if __name__ == "__main__":
    raise SystemExit(main())
