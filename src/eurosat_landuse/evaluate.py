from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime

from .config import describe_config, load_config
from .data import EUROSAT_CLASSES, build_dataloaders
from .model import build_model, count_parameters
from .paths import ensure_directory, project_path
from .runtime import select_device, set_seed
from .utils import module_available


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description="Evaluate an EfficientFormerV2 EuroSAT checkpoint.",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint created by train.py.")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Dataset split to evaluate.")
    parser.add_argument("--download", action="store_true", help="Download EuroSAT if it is not present.")
    parser.add_argument("--device", default="auto", help="Evaluation device: auto, cpu, mps, or cuda.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for evaluation.")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches for quick evaluation.")
    parser.add_argument("--run-name", default=None, help="Name for the saved evaluation metrics JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    print(describe_config(config))
    print(f"- torch installed: {module_available('torch')}")
    results_path = run_evaluation(
        config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        download=args.download,
        device_name=args.device,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        run_name=args.run_name,
    )
    print(f"- evaluation metrics saved to: {results_path}")
    return 0


def run_evaluation(
    config,
    *,
    checkpoint_path: str,
    split: str,
    download: bool,
    device_name: str,
    batch_size: int | None,
    max_batches: int | None,
    run_name: str | None,
) -> str:
    torch = __import__("torch")
    set_seed(config)
    device = select_device(device_name)
    print(f"- selected device: {device}")

    loaders = build_dataloaders(config, download=download, batch_size=batch_size)
    model = build_model(config).to(device)
    checkpoint = torch.load(project_path(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"- loaded checkpoint: {checkpoint_path}")
    print(f"- checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"- checkpoint val_accuracy: {checkpoint.get('val_accuracy', 'unknown')}")
    print(f"- model parameters: {count_parameters(model)}")

    training_cfg = dict(config.get("training", {}))
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=float(training_cfg.get("label_smoothing", 0.0)))
    metrics = evaluate_model(
        model,
        loaders[split],
        criterion,
        device,
        num_classes=int(dict(config.get("model", {})).get("num_classes", 10)),
        max_batches=max_batches,
    )
    print(f"- split: {split}")
    print(f"- loss: {metrics['loss']:.4f}")
    print(f"- accuracy: {metrics['accuracy']:.4f}")
    return save_evaluation_metrics(config, run_name, split, checkpoint_path, metrics)


def evaluate_model(model, dataloader, criterion, device, *, num_classes: int, max_batches: int | None) -> dict:
    torch = __import__("torch")
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

            for label, prediction in zip(labels.cpu().tolist(), predictions.cpu().tolist(), strict=True):
                class_total[label] += 1
                if label == prediction:
                    class_correct[label] += 1

    if total_samples == 0:
        raise RuntimeError("No samples were processed. Check the dataloader or max batch settings.")

    per_class = []
    for index in range(num_classes):
        total = class_total[index]
        per_class.append(
            {
                "class_index": index,
                "class_name": EUROSAT_CLASSES[index] if index < len(EUROSAT_CLASSES) else str(index),
                "correct": class_correct[index],
                "total": total,
                "accuracy": class_correct[index] / total if total else None,
            }
        )

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "total_samples": total_samples,
        "per_class": per_class,
    }


def save_evaluation_metrics(config, run_name: str | None, split: str, checkpoint_path: str, metrics: dict) -> str:
    outputs_cfg = dict(config.get("outputs", {}))
    metrics_dir = ensure_directory(project_path(outputs_cfg.get("metrics", "outputs/metrics")))
    resolved_run_name = run_name or datetime.now().strftime(f"eval_{split}_%Y%m%d_%H%M%S")
    metrics_path = metrics_dir / f"{resolved_run_name}.json"
    payload = {
        "run_name": resolved_run_name,
        "split": split,
        "checkpoint_path": checkpoint_path,
        "metrics": metrics,
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(metrics_path)


if __name__ == "__main__":
    raise SystemExit(main())
