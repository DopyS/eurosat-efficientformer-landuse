from __future__ import annotations

import argparse
from collections.abc import Sequence

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
        print("Use --smoke-test to run one forward/backward batch before full training.")
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


if __name__ == "__main__":
    raise SystemExit(main())
