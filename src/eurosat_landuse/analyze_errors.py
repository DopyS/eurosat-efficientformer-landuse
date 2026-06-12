from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .data import EUROSAT_CLASSES
from .paths import ensure_directory, project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze-errors",
        description="Create a Markdown error analysis report from an evaluation JSON file.",
    )
    parser.add_argument("--eval-json", required=True, help="Path to an evaluation JSON file.")
    parser.add_argument("--output", default=None, help="Output Markdown path.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of weak classes and confusions to include.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    eval_path = project_path(args.eval_json)
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    report = build_error_report(payload, top_k=args.top_k)
    output_path = resolve_output_path(args.output, payload)
    ensure_directory(output_path.parent)
    output_path.write_text(report, encoding="utf-8")
    print(f"- error analysis saved to: {output_path}")
    return 0


def resolve_output_path(output: str | None, payload: dict) -> Path:
    if output:
        return project_path(output)
    run_name = payload.get("run_name", "evaluation")
    return project_path("outputs", "metrics", f"{run_name}_error_analysis.md")


def build_error_report(payload: dict, *, top_k: int) -> str:
    metrics = payload["metrics"]
    per_class = metrics["per_class"]
    matrix = metrics["confusion_matrix"]
    labels = class_labels(len(matrix))
    weak_classes = sorted(per_class, key=lambda row: row["accuracy"] if row["accuracy"] is not None else -1)[:top_k]
    confusions = top_confusions(matrix, labels, top_k=top_k)

    lines = [
        f"# Error Analysis: {payload.get('run_name', 'evaluation')}",
        "",
        "## Overall Metrics",
        "",
        f"- Split: `{payload.get('split', 'unknown')}`",
        f"- Checkpoint: `{payload.get('checkpoint_path', 'unknown')}`",
        f"- Samples: {metrics['total_samples']}",
        f"- Loss: {metrics['loss']:.4f}",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        "",
        "## Weakest Classes",
        "",
        "| Class | Correct | Total | Accuracy | Error Rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in weak_classes:
        accuracy = row["accuracy"] if row["accuracy"] is not None else 0.0
        lines.append(
            f"| {row['class_name']} | {row['correct']} | {row['total']} | "
            f"{accuracy:.4f} | {1.0 - accuracy:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Top Confusions",
            "",
            "| True Class | Predicted Class | Count | Share of True Class |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for item in confusions:
        lines.append(
            f"| {item['true_class']} | {item['predicted_class']} | {item['count']} | "
            f"{item['share']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Report-ready Notes",
            "",
            "- The weakest classes should be discussed together with the confusion matrix rather than only with accuracy.",
            "- High confusion counts usually indicate visually similar land-use patterns or insufficient training coverage.",
            "- For a stronger final conclusion, rerun this analysis after longer training and compare whether the same classes remain weak.",
            "",
        ]
    )
    return "\n".join(lines)


def top_confusions(matrix: list[list[int]], labels: list[str], *, top_k: int) -> list[dict]:
    items = []
    for true_index, row in enumerate(matrix):
        total = sum(row)
        for predicted_index, count in enumerate(row):
            if true_index == predicted_index or count == 0:
                continue
            items.append(
                {
                    "true_class": labels[true_index],
                    "predicted_class": labels[predicted_index],
                    "count": count,
                    "share": count / total if total else 0.0,
                }
            )
    return sorted(items, key=lambda item: item["count"], reverse=True)[:top_k]


def class_labels(count: int) -> list[str]:
    return [EUROSAT_CLASSES[index] if index < len(EUROSAT_CLASSES) else str(index) for index in range(count)]


if __name__ == "__main__":
    raise SystemExit(main())
