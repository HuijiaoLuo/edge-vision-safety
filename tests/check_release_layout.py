from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "arduino_app/config.example.yaml",
    "arduino_app/python/helmet_classifier.py",
    "arduino_app/python/rider_identity.py",
    "arduino_app/python/register_rider_from_dataset.py",
    "arduino_app/python/evaluate_saved_frame.py",
    "arduino_app/python/safety_logic.py",
    "arduino_app/models/README.md",
    "arduino_app/users/README.md",
    "helmet_cv/prepare_helmet_crops.py",
    "helmet_cv/train_helmet_classifier.py",
    "helmet_cv/export_checkpoint_to_onnx.py",
    "helmet_cv/evaluate_frame.py",
    "helmet_cv/optional/add_public_yolo_to_existing_crops.py",
    "examples/masked_demo.jpg",
    "examples/masked_demo.json",
]

PRIVATE_PATTERNS = [
    "*.npy",
    "*.pt",
    "*.pth",
    "*.onnx",
]


def check_required_files_exist():
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    assert not missing, f"Missing required files: {missing}"


def check_private_artifacts_not_committed():
    leaked = []
    for pattern in PRIVATE_PATTERNS:
        leaked.extend(path for path in ROOT.rglob(pattern) if ".git" not in path.parts)
    assert not leaked, f"Private/model artifacts should not be committed: {leaked}"


def main():
    check_required_files_exist()
    check_private_artifacts_not_committed()
    print("Release layout check passed.")


if __name__ == "__main__":
    main()
