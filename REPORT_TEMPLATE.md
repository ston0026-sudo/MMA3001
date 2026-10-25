# Automated Detection of Pork-Rasher Packaging and Product-Placement Defects

**Course:** MMA3001 Numerical Methods and Machine Learning  
**Student:** [Your name and ID]  
**Date:** [Submission date]

> **Important:** This is a report template, not a completed empirical report. Replace every bracketed instruction with your own measured results and remove this notice before submission. Do not invent metrics or claim experiments that were not run.

## Abstract
This project investigates object detection for packaged pork-rasher images. A pretrained YOLO detector is used as a baseline and is compared with a version fine-tuned on the supplied Roboflow dataset. The models are evaluated on a held-out test split using precision, recall, mean average precision (mAP) and inference timing. The fine-tuned model achieved [insert measured results]. These results indicate [interpretation], subject to limitations associated with dataset size, annotation quality and differences between dataset images and real production conditions.

## 1. Engineering problem and context
Food-packaging operations require consistent inspection of product placement and packaging presentation. Manual inspection can be repetitive and may vary between operators. Computer vision can potentially assist by localising annotated objects in images, providing an auditable indication of where a detector believes relevant objects occur.

The chosen scope is **object detection of the classes explicitly annotated in Dataset 3**, rather than general food-safety assessment or an unrestricted judgement of meat quality. The model's capability is bounded by the dataset's class definitions and annotation conventions. A detection is not proof that food is safe or unsafe.

The intended use is a prototype decision-support component for further engineering evaluation, not a validated production-line safety system.

## 2. Inputs and outputs
### Inputs
- RGB images from the Dataset 3 Roboflow project, exported in YOLO format.
- `data.yaml`, containing class names and paths to training, validation and test data.
- Pretrained YOLO weights (`yolov8n.pt`) for the baseline and fine-tuning initialisation.

Images are read by the Ultralytics pipeline and resized/letterboxed to [actual image size]. The annotation format represents each object using a class identifier and normalised bounding-box centre coordinates, width and height.

### Outputs
For each input image, the detector returns zero or more bounding boxes, class labels and confidence scores. Evaluation returns precision, recall, mAP@0.5 and mAP@0.5:0.95 when available. A separate timing script estimates inference latency.

Missing files, malformed dataset paths or a missing test split should be treated as errors and fixed before evaluation. Empty label files may represent negative images; verify dataset conventions before treating them as invalid.

## 3. Computational method
### 3.1 Baseline
The baseline is the pretrained [exact model and version] detector evaluated on the test split without fine-tuning. It provides a reference for how well generic pretrained features transfer to this dataset.

### 3.2 Selected method
The selected method fine-tunes the same pretrained detector on the training split. Training settings were: epochs = [actual], image size = [actual], batch size = [actual], random seed = [actual], device = [CPU/GPU and model]. The validation split was used for training monitoring and settings decisions. The test split was held out for final evaluation.

YOLO is a single-stage object detector: it predicts bounding boxes, object confidence and class scores from an image in one forward pass. Fine-tuning updates model parameters using labelled examples so predictions become more appropriate to the target classes. Cite the official documentation or course material used to support this description.

### 3.3 Assumptions and parameters
- The dataset annotations consistently represent the intended objects and defects.
- Training and test images are sufficiently representative of the intended application.
- Train/validation/test splits are independent enough for the evaluation to be meaningful.
- Confidence threshold = [actual], image size = [actual].
- Model weights, package versions, seed and training settings are recorded for repeatability.

## 4. Alternative solutions and comparison
The main alternative is the un-fine-tuned pretrained model. Both approaches use the same architecture family, input size and held-out test split, helping isolate the effect of fine-tuning. A simpler traditional image-processing pipeline could also be considered, but may be sensitive to illumination, reflections and packaging variation; it was not implemented in this project [retain only if true].

| Method | Training on target data? | Test precision | Test recall | mAP@0.5 | mAP@0.5:0.95 | Mean inference time/image |
|---|---:|---:|---:|---:|---:|---:|
| Pretrained baseline | No | [ ] | [ ] | [ ] | [ ] | [ ] |
| Fine-tuned detector | Yes | [ ] | [ ] | [ ] | [ ] | [ ] |

Discuss the observed trade-off rather than assuming fine-tuning must improve every metric.

## 5. Validation methodology
The test split was used to assess generalisation to held-out images. Precision measures the fraction of predicted detections that are correct; recall measures the fraction of annotated objects that are detected. mAP aggregates precision-recall performance over confidence thresholds and, for mAP@0.5:0.95, multiple intersection-over-union thresholds.

Report the number of test images and the class-wise results if available. Inspect representative true positives, false positives and false negatives. State whether model selection used validation data and confirm that test results were not used to tune the final model.

This validation does not establish performance on all factories, cameras, packaging materials or operating conditions. If images are augmented versions of originals or near-duplicates cross the split boundaries, performance may be optimistic.

## 6. Optimisation and performance
Inference was profiled on [hardware, operating system, Python and library versions]. The model was warmed up for [n] images and timed over [n] test images at [image size]. The measured mean and median latencies were [values]. These measurements include [state what the timer includes] and are hardware-dependent.

The selected optimisation/change was [e.g. image-size comparison, batch-size comparison, or fine-tuning]. Compare accuracy and runtime before and after the change. Do not claim real-time suitability without specifying a target frame rate, end-to-end latency, hardware and measurement procedure.

## 7. Results and discussion
[Insert a results table and at least one relevant plot or prediction example generated by your experiments.]

The fine-tuned model [did/did not] outperform the baseline on [metrics]. Likely explanations include [supported by observed errors, not speculation alone]. Error inspection showed [specific observed patterns]. If results were weak, discuss plausible causes such as limited training data, class imbalance, annotation ambiguity, reflections, occlusion or overfitting.

## 8. Limitations and future work
Limitations include the dataset's scope, annotation quality, possible distribution shift, dependence on confidence threshold, compute constraints and lack of independent factory-line validation. Future work could collect more diverse images, audit annotations, evaluate per-class performance, test additional detector sizes, measure end-to-end throughput and assess robustness to lighting and occlusion.

## 9. Conclusion
This project implemented and evaluated [actual system] for [actual annotated task]. Compared with the baseline, the selected approach [actual conclusion based on measured metrics]. The evidence supports [appropriately limited conclusion], but further validation is required before operational deployment.

## References
- Roboflow Universe, “Pork Rasher Error Packaging” dataset, https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging (accessed [date]).
- Ultralytics, “YOLO Documentation,” https://docs.ultralytics.com/ (accessed [date]).
- [Add relevant course notes or academic references actually consulted.]
