"""Fine-tune a small pretrained YOLO detector on the training split."""
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Dataset YAML path")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained model weights")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="", help="e.g. 0 for CUDA GPU; blank lets Ultralytics choose")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device or None, seed=args.seed,
                project="runs", name="train", exist_ok=False,
                plots=True, patience=8, workers=2)
    print("Training complete. Record the actual output directory and settings in the report.")

if __name__ == "__main__":
    main()
