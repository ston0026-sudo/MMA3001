Pork Rasher Packaging Defect Detection
Unit: MMA3001 — Numerical Methods and Machine Learning
Project: Automated detection of pork-rasher packaging and product-placement defects

1. Project overview
This project investigates whether fine-tuning a pretrained YOLOv8 object-detection model can improve the detection of annotated pork-rasher packaging and product-placement defects.

The dataset is exported from the Roboflow Pork Rasher Error Packaging project in YOLO format.

The system accepts an image and predicts bounding boxes, class labels and confidence scores for objects represented by the dataset annotations.

Important scope limitation: this is a prototype for detecting annotated visual categories. It is not a food-safety, freshness or consumption-suitability assessment.

2. Dataset
The exported dataset contains three splits:

Split	    Images	   Labelled objects
Training	3,349	   2,923
Validation	120	       121
Test	    80	       79
Images have dimensions of 720 × 540 pixels. The model is trained and evaluated with an input size of 640 pixels.

The five annotation classes are:
- loose-meat
- packaging-error
- twisted-meat
- unsealed
- wrinkle

The dataset is imbalanced. The wrinkle class has only 18 training objects and no labelled examples in the validation or test splits. Therefore, its generalisation performance cannot be evaluated from the supplied split.

Some images have empty label files. These may represent negative images rather than annotation errors and should not be removed without verification.

3. Repository structure
pork-rasher-defect-detection/
├── README.md
├── REPORT_TEMPLATE.md
├── requirements.txt
├── data/
│   └── data.yaml
├── src/
│   ├── inspect_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── profile_inference.py
└── outputs/
    ├── train_25_epochs/
    ├── test_evaluation_25_epochs/
    ├── predictions_25_epochs/
    └── performance_profile.json
   
Generated model weights and the dataset itself may be stored separately because of their size. The paths above describe the intended project organisation; update them if your submitted repository uses a different layout.

5. Methods and experiments
The project uses Ultralytics YOLOv8n, a compact single-stage object detector.

Two fine-tuning experiments were conducted using pretrained YOLOv8n weights:

Experiment A: 5 training epochs.

Experiment B: 25 training epochs.

Both experiments used a 640-pixel image size and random seed 42. Training used a Tesla T4 GPU in Google Colab. The 25-epoch experiment used batch size 16 and a patience setting of 8.

A separate diagnostic evaluation ran the generic pretrained COCO model without fine-tuning. Its class definitions differ from the pork-rasher dataset classes, so its metrics are not a semantically equivalent baseline and must be interpreted with caution.

The validation split was used to monitor model performance. The test split was evaluated during development, so it should not be described as having been used only once.

5. Main test results
Metric	    5-epoch model	25-epoch model
Precision	    55.42%	    63.72%
Recall	        49.48%	    62.18%
mAP@0.5	        34.35%	    60.27%
mAP@0.5:0.95	19.11%	    33.60%

The 25-epoch experiment achieved higher values on all four reported aggregate metrics. Its test mAP@0.5 was 25.92 percentage points higher than the 5-epoch model's result.

These results indicate improved performance on this test split, not proven performance on all production environments. The dataset is small in validation and test size, has substantial class imbalance, and does not provide held-out examples for the wrinkle class.

6. Inference performance
The dedicated inference profiler measured the 25-epoch model on 80 test images using a Tesla T4 GPU.

Image size: 640 pixels

Warm-up images: 5

Mean latency: approximately 10.57 ms per image

Median latency: approximately 10.25 ms per image

Approximate throughput: 94.6 images per second

The timer includes Python/API overhead and is hardware-dependent. These measurements do not establish end-to-end production-line performance.

7. Running the code
Use Python with the dependencies listed in requirements.txt. A GPU is recommended for training.

Inspect the dataset:

python src/inspect_dataset.py --data data/data.yaml
Train a model:

python src/train.py --data data/data.yaml --model yolov8n.pt --epochs 25 --imgsz 640 --batch 16
Evaluate a model on the test split:
python src/evaluate.py --data data/data.yaml --weights path/to/best.pt --name final

Generate predictions:
python src/predict.py --weights path/to/best.pt --source data/test/images

Profile inference:
python src/profile_inference.py --weights path/to/best.pt --data data/data.yaml --imgsz 640 --max-images 80 --warmup 5

Use the actual weight-file path produced by your training run. Dataset paths in data/data.yaml must resolve correctly from the dataset root.

8. Limitations and future work
The wrinkle class cannot be validated because it has no examples in validation or test.

The class distribution is imbalanced, and some classes have very few annotated instances.

The independence of the dataset splits and possible overlap between augmented versions should be verified.

The generic pretrained model uses different class definitions, limiting the interpretation of its baseline metrics.

Performance on unseen cameras, packaging types, lighting conditions and real production lines has not been established.

Further work could include more representative labelled data, class-specific error analysis, controlled comparisons of training settings and evaluation on an independently collected dataset.

9. Reproducibility
Record the dataset version, Python and library versions, training parameters, random seed, weight-file location and evaluation settings. Preserve the generated metrics and figures used in the technical report.

Do not claim experiments, validation procedures or performance results that were not actually measured.
