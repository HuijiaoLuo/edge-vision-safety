# Eurosensors Edge AI

Clean portfolio release of my contribution to the Eurosensors 2026 Hackathon.

This repository contains the helmet model-training pipeline and edge-runtime logic for a scooter safety lock. Organizer-provided Arduino App Lab starter files, third-party assets, participant images, trained face embeddings, and model binaries are not included.

## Scope and Provenance

The source files under `helmet_cv/` and `arduino_app/python/` are my original or modified contribution. The repository presents the reusable training pipeline, inference modules, and safety decision logic; it does not claim that the complete hackathon application was developed from scratch.

The final system was validated on-device under live rider, helmet, and multi-person test conditions. The Arduino App Lab scaffold and other organizer-provided files are intentionally excluded.

## What It Does

The scooter remains locked unless:

```text
exactly one rider + registered rider identity + helmet detected
```

## Layout

```text
helmet_cv/                 Final helmet training, export, and evaluation scripts
helmet_cv/optional/        Optional public-data fine-tuning comparison
arduino_app/python/        Runtime inference modules and starter-free safety logic
arduino_app/models/        Model download/export notes
arduino_app/users/         Placeholder for private registered-rider embeddings
examples/                  Privacy-masked demo image/json
tests/                     Lightweight release checks
.github/workflows/ci.yml   Linux CI
```

## Final Helmet Pipeline

Prepare local private crops:

```bash
python helmet_cv/prepare_helmet_crops.py --input-dir TRAIN_READY dataset_helmet --output-dir helmet_cv/dataset_updated_clean
```

Train EfficientNet-B0:

```bash
python helmet_cv/train_helmet_classifier.py --data-dir helmet_cv/dataset_updated_clean --model efficientnet_b0 --output-dir helmet_cv/runs/efficientnet_b0_updated_clean --epochs 30 --batch-size 8 --unfreeze-backbone --lr 0.0001
```

Export ONNX:

```bash
python helmet_cv/export_checkpoint_to_onnx.py --model-path helmet_cv/runs/efficientnet_b0_updated_clean/helmet_classifier.pt --output arduino_app/models/helmet_classifier.onnx
```

Final augmentation: crop jitter, horizontal flip, brightness/contrast/color jitter, small rotation, and ImageNet normalization. Helmet-color recoloring was an early experiment and is not included in this release pipeline.

## Optional Public-Data Experiment

Public dataset used for comparison:

```text
Bike Helmet Detection v2, YOLOv8 format, Public Domain
https://universe.roboflow.com/santhu-ajspn/bike-helmet-detection-2vdjo-ahcew/dataset/2
```

Append sampled public crops to training only, keeping local validation unchanged:

```bash
python helmet_cv/optional/add_public_yolo_to_existing_crops.py --existing-crop-dir helmet_cv/dataset_updated_clean --public-yolo-dir "Bike Helmet Detection.v2i.yolov8" --output-dir helmet_cv/dataset_updated_public_local --max-public-per-class 200
```

## Arduino Runtime

Runtime model files are expected at:

```text
/app/models/helmet_classifier.onnx
/app/models/face_recognition.onnx
/app/models/face_detection_yunet.onnx
/app/users/*.npy
```

`safety_logic.py` contains the starter-free frame evaluation logic. In the hackathon app, this was called from the Arduino App Lab camera loop.

## Privacy

Raw participant images and `users/*.npy` face embeddings are intentionally excluded. The included demo image has the face region masked.

The tracked demo JPEG was checked for EXIF, XMP, IPTC, and JPEG comment metadata. It contains none, and the accompanying JSON uses only the neutral label `masked_demo_rider`; original filenames, names, and team information are not retained.

## License

The Apache-2.0 license in `LICENSE` applies only to original source code authored or modified for this release. It does not relicense organizer-provided files, third-party assets or models, public datasets, participant images, or biometric embeddings; those materials are excluded or remain subject to their respective terms.

## CI

GitHub Actions runs on Linux for every push and pull request. It performs syntax checks and verifies that private/model artifacts were not committed.
