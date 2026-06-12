from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime

from .config import describe_config, load_config
from .data import build_data_pipeline, build_dataloaders
from .model import build_model, build_model_summary, count_parameters
from .paths import ensure_directory, project_path
from .runtime import select_device, set_seed
from .utils import module_available


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train",
        description="Training entrypoint for the EuroSAT project scaffold.",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    parser.add_argument("--show-config", action="store_true", help="Print the loaded config and exit.")
    parser.add_argument("--smoke-test", action="store_true", help="Run one forward/backward batch only.")
    parser.add_argument("--download", action="store_true", help="Download EuroSAT if it is not present.")
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, mps, or cuda.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for this run.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit train batches for quick runs.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Limit validation batches for quick runs.")
    parser.add_argument("--run-name", default=None, help="Name for outputs under outputs/metrics and outputs/checkpoints.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    if args.show_config:
        print(describe_config(config))
        return 0

    ensure_directory(project_path("outputs"))
    print(describe_config(config))
    print("Training scaffold is ready.")
    print(f"- torch installed: {module_available('torch')}")
    print(f"- torchvision installed: {module_available('torchvision')}")
    print(f"- timm installed: {module_available('timm')}")
    print(f"- planned data pipeline: {build_data_pipeline(config)}")
    print(f"- planned model summary: {build_model_summary(config)}")
    if args.smoke_test:
        run_smoke_test(config, download=args.download, device_name=args.device, batch_size=args.batch_size)
    else:
        run_training(
            config,
            download=args.download,
            device_name=args.device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
            run_name=args.run_name,
        )
    return 0


def run_smoke_test(config, *, download: bool, device_name: str, batch_size: int | None) -> None:
    torch = __import__("torch")
    set_seed(config)
    device = select_device(device_name)
    print(f"- selected device: {device}")

    loaders = build_dataloaders(config, download=download, batch_size=batch_size)
    images, labels = next(iter(loaders["train"]))
    images = images.to(device)
    labels = labels.to(device)

    model = build_model(config).to(device)
    model.train()
    parameter_counts = count_parameters(model)
    print(f"- model parameters: {parameter_counts}")

    training_cfg = dict(config.get("training", {}))
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=float(training_cfg.get("label_smoothing", 0.0)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 0.0003)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0001)),
    )

    optimizer.zero_grad(set_to_none=True)
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    predictions = outputs.argmax(dim=1)
    accuracy = (predictions == labels).float().mean().item()
    print(f"- smoke batch image shape: {tuple(images.shape)}")
    print(f"- smoke logits shape: {tuple(outputs.shape)}")
    print(f"- smoke loss: {loss.item():.4f}")
    print(f"- smoke accuracy: {accuracy:.4f}")
    print("Training smoke test passed.")


def run_training(
    config,
    *,
    download: bool,
    device_name: str,
    batch_size: int | None,
    epochs: int | None,
    max_train_batches: int | None,
    max_val_batches: int | None,
    run_name: str | None,
) -> None:
    torch = __import__("torch")
    set_seed(config)
    device = select_device(device_name)
    print(f"- selected device: {device}")

    loaders = build_dataloaders(config, download=download, batch_size=batch_size)
    model = build_model(config).to(device)
    print(f"- model parameters: {count_parameters(model)}")

    training_cfg = dict(config.get("training", {}))
    total_epochs = int(epochs or training_cfg.get("epochs", 10))
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=float(training_cfg.get("label_smoothing", 0.0)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 0.0003)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0001)),
    )
    mixup_alpha = float(training_cfg.get("mixup_alpha", 0.0))
    print(f"- mixup_alpha: {mixup_alpha}")

    resolved_run_name = run_name or datetime.now().strftime("baseline_%Y%m%d_%H%M%S")
    metrics_history = []
    best_val_accuracy = -1.0
    best_checkpoint_path = None

    for epoch in range(1, total_epochs + 1):
        train_metrics = run_one_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            max_batches=max_train_batches,
            mixup_alpha=mixup_alpha,
        )
        val_metrics = run_one_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            optimizer=None,
            max_batches=max_val_batches,
            mixup_alpha=0.0,
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        metrics_history.append(epoch_metrics)
        print(
            f"epoch {epoch}/{total_epochs} "
            f"train_loss={epoch_metrics['train_loss']:.4f} "
            f"train_acc={epoch_metrics['train_accuracy']:.4f} "
            f"val_loss={epoch_metrics['val_loss']:.4f} "
            f"val_acc={epoch_metrics['val_accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_checkpoint_path = save_checkpoint(model, config, resolved_run_name, epoch, val_metrics["accuracy"])

    metrics_path = save_metrics(config, resolved_run_name, metrics_history, best_checkpoint_path)
    print(f"- best val accuracy: {best_val_accuracy:.4f}")
    print(f"- metrics saved to: {metrics_path}")
    if best_checkpoint_path:
        print(f"- best checkpoint saved to: {best_checkpoint_path}")


def run_one_epoch(
    model,
    dataloader,
    criterion,
    device,
    *,
    optimizer=None,
    max_batches: int | None = None,
    mixup_alpha: float = 0.0,
) -> dict[str, float]:
    torch = __import__("torch")
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch_idx, (images, labels) in enumerate(dataloader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
            if is_training and mixup_alpha > 0:
                mixed_images, labels_a, labels_b, lam = apply_mixup(images, labels, mixup_alpha)
                outputs = model(mixed_images)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                hard_labels = labels
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                hard_labels = labels
            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == hard_labels).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("No samples were processed. Check the dataloader or max batch settings.")
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def apply_mixup(images, labels, alpha: float):
    torch = __import__("torch")
    if alpha <= 0:
        return images, labels, labels, 1.0
    beta = torch.distributions.Beta(alpha, alpha)
    lam = float(beta.sample().item())
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1 - lam) * images[index]
    return mixed_images, labels, labels[index], lam


def save_checkpoint(model, config, run_name: str, epoch: int, val_accuracy: float):
    torch = __import__("torch")
    outputs_cfg = dict(config.get("outputs", {}))
    checkpoint_dir = ensure_directory(project_path(outputs_cfg.get("checkpoints", "outputs/checkpoints")))
    checkpoint_path = checkpoint_dir / f"{run_name}_best.pt"
    torch.save(
        {
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "model_state_dict": model.state_dict(),
            "config": config,
        },
        checkpoint_path,
    )
    return checkpoint_path


def save_metrics(config, run_name: str, history: list[dict[str, float]], best_checkpoint_path) -> str:
    outputs_cfg = dict(config.get("outputs", {}))
    metrics_dir = ensure_directory(project_path(outputs_cfg.get("metrics", "outputs/metrics")))
    metrics_path = metrics_dir / f"{run_name}.json"
    payload = {
        "run_name": run_name,
        "history": history,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path else None,
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(metrics_path)


if __name__ == "__main__":
    raise SystemExit(main())
