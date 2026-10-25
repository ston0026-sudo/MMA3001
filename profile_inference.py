"""Measure approximate per-image inference latency on test images."""
import argparse, json, time
from pathlib import Path
import yaml
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    root = Path(cfg.get("path", Path(args.data).resolve().parent))
    if not root.is_absolute():
        root = (Path(args.data).resolve().parent / root).resolve()
    test_path = Path(cfg["test"])
    if not test_path.is_absolute():
        test_path = (root / test_path).resolve()
    if test_path.is_file() and test_path.suffix == ".txt":
        images = [Path(x.strip()) for x in test_path.read_text().splitlines() if x.strip()]
        images = [x if x.is_absolute() else test_path.parent / x for x in images]
    else:
        images = sorted(x for x in test_path.rglob("*") if x.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"})
    images = images[:args.max_images]
    if not images:
        raise FileNotFoundError(f"No test images found at {test_path}")
    model = YOLO(args.weights)
    for p in images[:args.warmup]:
        model.predict(str(p), imgsz=args.imgsz, verbose=False)
    durations = []
    for p in images:
        t0 = time.perf_counter()
        model.predict(str(p), imgsz=args.imgsz, verbose=False)
        durations.append(time.perf_counter() - t0)
    result = {
        "weights": args.weights, "test_path": str(test_path), "images_timed": len(durations),
        "imgsz": args.imgsz, "warmup_images": min(args.warmup, len(images)),
        "mean_seconds_per_image": sum(durations)/len(durations),
        "median_seconds_per_image": sorted(durations)[len(durations)//2],
        "approx_images_per_second": len(durations)/sum(durations),
        "note": "Includes Python/API overhead; hardware-specific; not a production-line guarantee."
    }
    Path("outputs").mkdir(exist_ok=True)
    out = Path("outputs/performance_profile.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
