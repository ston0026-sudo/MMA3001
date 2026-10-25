"""Run inference on an image, folder or video and save annotated outputs."""
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    model = YOLO(args.weights)
    model.predict(source=args.source, imgsz=args.imgsz, conf=args.conf,
                  save=True, project="runs/predict", name="pork_rasher")
    print("Predictions saved under runs/predict/.")

if __name__ == "__main__":
    main()
