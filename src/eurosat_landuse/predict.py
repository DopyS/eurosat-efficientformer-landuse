from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .config import describe_config, load_config
from .data import EUROSAT_CLASSES, build_transforms
from .model import build_model
from .paths import project_path
from .runtime import select_device
from .utils import module_available


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict",
        description="Prediction entrypoint for the EuroSAT project scaffold.",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint created by train.py.")
    parser.add_argument("--image", required=True, help="Path to one image for prediction.")
    parser.add_argument("--device", default="auto", help="Prediction device: auto, cpu, mps, or cuda.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to print.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    print(describe_config(config))
    print(f"- streamlit installed: {module_available('streamlit')}")
    predictions = predict_image(
        config,
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        device_name=args.device,
        top_k=args.top_k,
    )
    for item in predictions:
        print(f"{item['rank']}. {item['class_name']} confidence={item['confidence']:.4f}")
    return 0


def load_checkpoint_model(config, checkpoint_path: str, *, device_name: str = "auto"):
    torch = __import__("torch")
    device = select_device(device_name)
    model = build_model(config).to(device)
    checkpoint = torch.load(project_path(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


def predict_image(
    config,
    *,
    checkpoint_path: str,
    image_path: str | Path,
    device_name: str = "auto",
    top_k: int = 3,
) -> list[dict]:
    pillow = __import__("PIL.Image").Image
    image = pillow.open(project_path(image_path)).convert("RGB")
    return predict_pil_image(
        config,
        checkpoint_path=checkpoint_path,
        image=image,
        device_name=device_name,
        top_k=top_k,
    )


def predict_pil_image(
    config,
    *,
    checkpoint_path: str,
    image,
    device_name: str = "auto",
    top_k: int = 3,
) -> list[dict]:
    torch = __import__("torch")
    model, device = load_checkpoint_model(config, checkpoint_path, device_name=device_name)
    transform = build_transforms(config, train=False)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        values, indices = torch.topk(probabilities, k=min(top_k, probabilities.numel()))

    predictions = []
    for rank, (value, index) in enumerate(zip(values.cpu().tolist(), indices.cpu().tolist(), strict=True), start=1):
        predictions.append(
            {
                "rank": rank,
                "class_index": index,
                "class_name": EUROSAT_CLASSES[index] if index < len(EUROSAT_CLASSES) else str(index),
                "confidence": value,
            }
        )
    return predictions


if __name__ == "__main__":
    raise SystemExit(main())
