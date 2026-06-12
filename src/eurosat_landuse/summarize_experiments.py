from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import load_config
from .paths import ensure_directory, project_path, project_root


FIELDNAMES = [
    "run_name",
    "record_type",
    "split",
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "eval_loss",
    "eval_accuracy",
    "total_samples",
    "checkpoint_path",
    "source_file",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="summarize-experiments",
        description="Summarize local experiment metrics into CSV and Markdown files.",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    parser.add_argument("--metrics-dir", default=None, help="Directory containing metrics JSON files.")
    parser.add_argument("--output-name", default="experiment_summary", help="Output filename stem.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    outputs_cfg = dict(config.get("outputs", {}))
    metrics_dir = project_path(args.metrics_dir or outputs_cfg.get("metrics", "outputs/metrics"))
    rows = collect_metric_rows(metrics_dir)
    if not rows:
        raise RuntimeError(f"No metrics JSON files found in {metrics_dir}.")

    csv_path = metrics_dir / f"{args.output_name}.csv"
    md_path = metrics_dir / f"{args.output_name}.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"- summarized {len(rows)} metric rows")
    print(f"- csv saved to: {csv_path}")
    print(f"- markdown saved to: {md_path}")
    return 0


def collect_metric_rows(metrics_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(metrics_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = parse_metric_payload(payload, path)
        if row:
            rows.append(row)
    return rows


def parse_metric_payload(payload: dict[str, Any], source_file: Path) -> dict[str, Any] | None:
    if "history" in payload:
        return parse_training_payload(payload, source_file)
    if "metrics" in payload:
        return parse_evaluation_payload(payload, source_file)
    return None


def parse_training_payload(payload: dict[str, Any], source_file: Path) -> dict[str, Any]:
    history = payload.get("history") or []
    final_epoch = history[-1] if history else {}
    return empty_row(
        run_name=payload.get("run_name", source_file.stem),
        record_type="train",
        source_file=source_file,
        epoch=final_epoch.get("epoch"),
        train_loss=final_epoch.get("train_loss"),
        train_accuracy=final_epoch.get("train_accuracy"),
        val_loss=final_epoch.get("val_loss"),
        val_accuracy=final_epoch.get("val_accuracy"),
        checkpoint_path=payload.get("best_checkpoint_path"),
    )


def parse_evaluation_payload(payload: dict[str, Any], source_file: Path) -> dict[str, Any]:
    metrics = payload.get("metrics") or {}
    return empty_row(
        run_name=payload.get("run_name", source_file.stem),
        record_type="eval",
        split=payload.get("split"),
        source_file=source_file,
        eval_loss=metrics.get("loss"),
        eval_accuracy=metrics.get("accuracy"),
        total_samples=metrics.get("total_samples"),
        checkpoint_path=payload.get("checkpoint_path"),
    )


def empty_row(**values: Any) -> dict[str, Any]:
    row = {field: "" for field in FIELDNAMES}
    row.update({key: format_value(value) for key, value in values.items() if key in row})
    return row


def format_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, Path):
        return format_path(value)
    if isinstance(value, str):
        return format_path(Path(value)) if value else ""
    return value


def format_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root()))
    except (OSError, ValueError):
        return str(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    visible_fields = [
        "run_name",
        "record_type",
        "split",
        "train_accuracy",
        "val_accuracy",
        "eval_accuracy",
        "total_samples",
    ]
    lines = [
        "# Experiment Summary",
        "",
        "| " + " | ".join(visible_fields) + " |",
        "| " + " | ".join(["---"] * len(visible_fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in visible_fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
