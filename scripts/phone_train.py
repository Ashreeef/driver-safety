import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from ultralytics import YOLO

# Add project root to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob("*.jpg")) + list(path.glob("*.jpeg")) + list(path.glob("*.png")))


def _read_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)


def get_phone_class_id(dataset_path: Path) -> int:
    cfg = _read_yaml(dataset_path / "data.yaml")
    names = cfg.get("names", [])

    if isinstance(names, dict):
        for k, name in names.items():
            if "phone" in str(name).lower():
                return int(k)
    else:
        for i, name in enumerate(names):
            if "phone" in str(name).lower():
                return i

    raise ValueError("Could not find a phone class in data.yaml")


def remove_augmented_train_copies(dataset_path: Path) -> int:
    img_dir = dataset_path / "train" / "images"
    lbl_dir = dataset_path / "train" / "labels"
    if not img_dir.exists():
        return 0

    aug_pattern = re.compile(
        r"_(aug|mosaic|flip|rotate|crop|bright|noise|blur|gray|hflip|vflip|shear|zoom)(?:_\d+|\d+)?$",
        re.IGNORECASE,
    )

    removed = 0
    for image_path in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png")):
        if not aug_pattern.search(image_path.stem):
            continue
        image_path.unlink(missing_ok=True)
        label_path = lbl_dir / f"{image_path.stem}.txt"
        label_path.unlink(missing_ok=True)
        removed += 1
    return removed


def download_dataset(dataset_path: Path, version: int) -> None:
    try:
        from dotenv import load_dotenv
        from roboflow import Roboflow
    except ImportError as exc:
        raise ImportError(
            "roboflow and python-dotenv are required. Install with: pip install roboflow python-dotenv"
        ) from exc

    load_dotenv(dotenv_path=Path(".env"))
    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY not found. Add it in a .env file at repository root.")

    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("pruebadeteccioncelulareschoferes").project("cellphone-detector-drivers-ma3vx")

    for fmt in ["yolov8", "yolov9", "yolov5pytorch"]:
        try:
            print(f"Trying download format={fmt}, version={version}...")
            project.version(version).download(fmt, location=str(dataset_path))
            print("Download succeeded")
            break
        except Exception as exc:
            print(f"Format {fmt} failed: {exc}")
    else:
        raise RuntimeError("Dataset download failed for all formats")

    removed = remove_augmented_train_copies(dataset_path)
    if removed:
        print(f"Removed {removed} augmented training images")


def build_phone_only_dataset(dataset_path: Path, output_path: Path) -> Path:
    phone_id = get_phone_class_id(dataset_path)
    print(f"Phone class id: {phone_id}")

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid", "test"]:
        src_img = dataset_path / split / "images"
        src_lbl = dataset_path / split / "labels"
        dst_img = output_path / split / "images"
        dst_lbl = output_path / split / "labels"

        if not src_img.exists() or not src_lbl.exists():
            continue

        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        for label_file in src_lbl.glob("*.txt"):
            lines = label_file.read_text(encoding="utf-8").splitlines()
            phone_lines: List[str] = []
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split()
                if int(parts[0]) == phone_id:
                    phone_lines.append("0 " + " ".join(parts[1:]))

            if split != "test" and not phone_lines:
                continue

            image_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = src_img / f"{label_file.stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path is None:
                continue

            shutil.copy2(image_path, dst_img / image_path.name)
            (dst_lbl / label_file.name).write_text("\n".join(phone_lines), encoding="utf-8")

    phone_yaml = {
        "path": str(output_path.resolve()),
        "train": str((output_path / "train" / "images").resolve()),
        "val": str((output_path / "valid" / "images").resolve()),
        "test": str((output_path / "test" / "images").resolve()),
        "nc": 1,
        "names": ["phone"],
    }
    yaml_path = output_path / "data.yaml"
    _write_yaml(yaml_path, phone_yaml)
    print(f"Phone-only dataset ready: {output_path}")
    return yaml_path


def train_model(
    data_yaml: Path,
    pretrained_model: str,
    project: str,
    experiment_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
):
    model = YOLO(pretrained_model)
    result = model.train(
        data=str(data_yaml),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        freeze=10,
        warmup_epochs=5,
        patience=30,
        close_mosaic=20,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        erasing=0.3,
        project=project,
        name=experiment_name,
        save_period=10,
        plots=True,
        verbose=True,
        seed=42,
    )
    best_model = Path(result.save_dir) / "weights" / "best.pt"
    print(f"Training complete. Best model: {best_model}")
    return best_model


def evaluate_model(model_path: Path, data_yaml: Path, split: str = "val") -> None:
    model = YOLO(str(model_path))
    metrics = model.val(data=str(data_yaml), imgsz=640, batch=16, split=split, verbose=True)
    print(f"{split} metrics:")
    print(f"  mAP@50: {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phone dataset preparation and YOLO training")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Download and prepare dataset")
    prep.add_argument("--dataset", type=str, default="dataset", help="Raw dataset directory")
    prep.add_argument("--dataset-version", type=int, default=1, help="Roboflow dataset version")
    prep.add_argument(
        "--mode",
        type=str,
        choices=["phone_only", "all_classes"],
        default="phone_only",
        help="phone_only builds a new one-class dataset; all_classes keeps original labels",
    )
    prep.add_argument("--phone-dataset", type=str, default="dataset_phone", help="Output path for phone-only dataset")

    train = sub.add_parser("train", help="Train phone detector")
    train.add_argument("--data", type=str, default="dataset_phone/data.yaml", help="Path to data.yaml")
    train.add_argument("--pretrained", type=str, default="yolov8n.pt", help="Pretrained model path")
    train.add_argument("--project", type=str, default="runs/phone_detection", help="Training output project")
    train.add_argument("--name", type=str, default="yolov8n_phone_v2", help="Experiment name")
    train.add_argument("--epochs", type=int, default=60)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--device", type=str, default="0")

    evaluate = sub.add_parser("eval", help="Evaluate a trained model")
    evaluate.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    evaluate.add_argument("--data", type=str, default="dataset_phone/data.yaml", help="Path to data.yaml")
    evaluate.add_argument("--split", type=str, choices=["val", "test"], default="val")

    args = parser.parse_args()

    if args.command == "prepare":
        dataset_path = Path(args.dataset)
        download_dataset(dataset_path, version=args.dataset_version)

        if args.mode == "phone_only":
            yaml_path = build_phone_only_dataset(dataset_path, Path(args.phone_dataset))
            print(f"Use this YAML for training: {yaml_path}")
        else:
            print(f"Use original YAML for training: {dataset_path / 'data.yaml'}")

        print(f"Train images in raw dataset: {_count_images(dataset_path / 'train' / 'images')}")

    elif args.command == "train":
        train_model(
            data_yaml=Path(args.data),
            pretrained_model=args.pretrained,
            project=args.project,
            experiment_name=args.name,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )

    elif args.command == "eval":
        evaluate_model(Path(args.weights), Path(args.data), split=args.split)


if __name__ == "__main__":
    main()
