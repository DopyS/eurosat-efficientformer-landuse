from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .config import describe_config, load_config
from .data import EUROSAT_CLASSES, build_dataloaders
from .paths import ensure_directory, project_path
from .predict import load_checkpoint_model
from .runtime import set_seed

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export-errors",
        description="Export representative misclassified EuroSAT samples for report figures.",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint created by train.py.")
    parser.add_argument("--split", choices=["val", "test"], default="test", help="Dataset split to inspect.")
    parser.add_argument("--true-class", required=True, help="True class name or index to export.")
    parser.add_argument("--predicted-class", required=True, help="Predicted class name or index to export.")
    parser.add_argument("--output-dir", default=None, help="Output directory for exported samples.")
    parser.add_argument("--device", default="auto", help="Inference device: auto, cpu, mps, or cuda.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for scanning samples.")
    parser.add_argument("--limit", type=int, default=12, help="Maximum number of samples to export.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional batch limit for quick checks.")
    parser.add_argument("--download", action="store_true", help="Download EuroSAT if it is not present.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    print(describe_config(config))
    output_dir = export_error_samples(
        config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        true_class=args.true_class,
        predicted_class=args.predicted_class,
        output_dir=args.output_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
        max_batches=args.max_batches,
        download=args.download,
    )
    print(f"- exported error samples to: {output_dir}")
    return 0


def export_error_samples(
    config,
    *,
    checkpoint_path: str,
    split: str,
    true_class: str,
    predicted_class: str,
    output_dir: str | None,
    device_name: str,
    batch_size: int,
    limit: int,
    max_batches: int | None,
    download: bool,
) -> Path:
    torch = __import__("torch")
    set_seed(config)
    true_index = resolve_class(true_class)
    predicted_index = resolve_class(predicted_class)
    resolved_output = ensure_directory(
        project_path(output_dir)
        if output_dir
        else project_path(
            "outputs",
            "error_samples",
            f"{split}_{EUROSAT_CLASSES[true_index]}_to_{EUROSAT_CLASSES[predicted_index]}",
        )
    )

    model, device = load_checkpoint_model(config, checkpoint_path, device_name=device_name)
    loaders = build_dataloaders(config, download=download, batch_size=batch_size)
    dataloader = loaders[split]
    records = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            for item_idx in range(labels.size(0)):
                label = int(labels[item_idx].item())
                prediction = int(predictions[item_idx].item())
                if label != true_index or prediction != predicted_index:
                    continue
                confidence = float(probabilities[item_idx, prediction].item())
                image = tensor_to_pil(images[item_idx].detach().cpu())
                filename = (
                    f"sample_{len(records) + 1:03d}_true-{EUROSAT_CLASSES[label]}"
                    f"_pred-{EUROSAT_CLASSES[prediction]}_conf-{confidence:.4f}.png"
                )
                path = resolved_output / filename
                image.save(path)
                records.append(
                    {
                        "file": filename,
                        "true_class": EUROSAT_CLASSES[label],
                        "predicted_class": EUROSAT_CLASSES[prediction],
                        "confidence": confidence,
                        "batch": batch_idx,
                        "batch_index": item_idx,
                    }
                )
                if len(records) >= limit:
                    break
            if len(records) >= limit:
                break

    write_index(resolved_output, records, checkpoint_path, split, true_index, predicted_index)
    if records:
        write_contact_sheet(resolved_output, records)
    return resolved_output


def resolve_class(value: str) -> int:
    if value.isdigit():
        index = int(value)
        if 0 <= index < len(EUROSAT_CLASSES):
            return index
    normalized = value.strip().lower()
    for index, class_name in enumerate(EUROSAT_CLASSES):
        if class_name.lower() == normalized:
            return index
    choices = ", ".join(EUROSAT_CLASSES)
    raise ValueError(f"Unknown class {value!r}. Use an index or one of: {choices}")


def tensor_to_pil(tensor):
    pillow = __import__("PIL.Image").Image
    torch = __import__("torch")
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    image = (tensor * std + mean).clamp(0, 1)
    image = (image * 255).round().to(dtype=torch.uint8).permute(1, 2, 0).numpy()
    return pillow.fromarray(image)


def write_index(
    output_dir: Path,
    records: list[dict],
    checkpoint_path: str,
    split: str,
    true_index: int,
    predicted_index: int,
) -> None:
    lines = [
        f"# Error Samples: {EUROSAT_CLASSES[true_index]} -> {EUROSAT_CLASSES[predicted_index]}",
        "",
        f"- Split: `{split}`",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Exported samples: {len(records)}",
        "",
        "| File | True Class | Predicted Class | Confidence | Batch | Batch Index |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            f"| `{record['file']}` | {record['true_class']} | {record['predicted_class']} | "
            f"{record['confidence']:.4f} | {record['batch']} | {record['batch_index']} |"
        )
    if not records:
        lines.extend(["", "No matching error samples were found."])
    (output_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_contact_sheet(output_dir: Path, records: list[dict], *, columns: int = 4) -> None:
    pillow_image = __import__("PIL.Image").Image
    pillow_draw = __import__("PIL.ImageDraw").ImageDraw
    images = [pillow_image.open(output_dir / record["file"]).convert("RGB") for record in records]
    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images) + 34
    rows = (len(images) + columns - 1) // columns
    sheet = pillow_image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = pillow_draw.Draw(sheet)

    for index, (image, record) in enumerate(zip(images, records, strict=True)):
        row = index // columns
        column = index % columns
        x = column * cell_width
        y = row * cell_height
        sheet.paste(image, (x, y))
        label = f"{index + 1}. conf={record['confidence']:.2f}"
        draw.text((x + 6, y + image.height + 8), label, fill=(20, 20, 20))

    sheet.save(output_dir / "contact_sheet.png")


if __name__ == "__main__":
    raise SystemExit(main())
