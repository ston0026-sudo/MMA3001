"""Evaluate a detector on the test split and save key metrics."""
import argparse, json, time
from pathlib import Path
from ultralytics import YOLO
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--name", default="evaluation")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    if not cfg.get("test"):
        raise ValueError("data.yaml has no 'test' split. Export a dataset with a test split.")
    model = YOLO(args.weights)
    start = time.perf_counter()
    metrics = model.val(data=args.data, split="test", imgsz=args.imgsz,
                        batch=args.batch, plots=True, project="runs/detect",
                        name=f"eval_{args.name}")
    elapsed = time.perf_counter() - start
    box = getattr(metrics, "box", None)
    result = {"name": args.name, "weights": args.weights, "data": args.data,
              "elapsed_seconds_including_validation": elapsed}
    if box is not None:
        for key in ("mp", "mr", "map50", "map"):
            value = getattr(box, key, None)
            if value is not None:
                result[{"mp":"precision","mr":"recall","map50":"mAP50","map":"mAP50_95"}[key]] = float(value)
        result["per_class_map"] = [float(x) for x in getattr(box, "maps", [])]
    Path("outputs").mkdir(exist_ok=True)
    output = Path("outputs") / f"{args.name}_metrics.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {output}")

if __name__ == "__main__":
    main()
