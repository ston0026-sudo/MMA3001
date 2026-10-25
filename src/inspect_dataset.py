"""Inspect a YOLO-format dataset and report split/image/label counts."""
from pathlib import Path
import argparse, json
import yaml
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    args = parser.parse_args()
    yaml_path = Path(args.data).resolve()
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()

    summary = {"yaml": str(yaml_path), "root": str(root), "names": cfg.get("names", {}),
               "nc": cfg.get("nc"), "splits": {}}
    for split in ("train", "val", "valid", "test"):
        value = cfg.get(split)
        if not value:
            continue
        p = Path(value)
        if not p.is_absolute():
            p = (root / p).resolve()
        # A YAML path can be a directory or a text file listing image paths.
        if p.is_file() and p.suffix.lower() == ".txt":
            images = [Path(line.strip()) for line in p.read_text().splitlines() if line.strip()]
            images = [x if x.is_absolute() else (p.parent / x).resolve() for x in images]
        elif p.is_dir():
            images = [x for x in p.rglob("*") if x.suffix.lower() in IMAGE_EXTS]
        else:
            images = []
            print(f"WARNING: split path not found: {split} -> {p}")
        label_dir = p.parent / "labels" if p.name == "images" else p.parent / "labels"
        labels = []
        for image_path in images:
            candidate = label_dir / f"{image_path.stem}.txt"
            # Standard YOLO directory layout: images/... -> labels/...
            parts = list(image_path.parts)
            if "images" in parts:
                parts[parts.index("images")] = "labels"
                candidate2 = Path(*parts).with_suffix(".txt")
                if candidate2.exists():
                    candidate = candidate2
            if candidate.exists():
                labels.append(candidate)
        dims = []
        for image_path in images[:100]:
            try:
                with Image.open(image_path) as im:
                    dims.append(im.size)
            except Exception:
                pass
        summary["splits"][split] = {
            "images": len(images), "label_files_found": len(labels),
            "sample_dimensions_first_100": sorted(set(dims)),
            "path": str(p)
        }
    Path("outputs").mkdir(exist_ok=True)
    out = Path("outputs/dataset_summary.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")

if __name__ == "__main__":
    main()
