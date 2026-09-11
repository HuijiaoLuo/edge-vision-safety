import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageOps


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def require_torch():
    """Import PyTorch dependencies only when frame evaluation is executed."""
    try:
        import torch
        from torchvision import models, transforms
    except ImportError as exc:
        raise SystemExit(
            "Missing PyTorch/torchvision. Install dependencies with:\n"
            "  python -m pip install -r helmet_cv/requirements.txt"
        ) from exc
    return torch, models, transforms


def clamp(value, low, high):
    """Clamp a coordinate to an image boundary."""
    return max(low, min(high, value))


def padded_bbox(bbox, image_size, padding):
    """Expand a JSON face/head box into the classifier ROI."""
    width, height = image_size
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    pad_x = int(round(w * padding))
    pad_y = int(round(h * padding))
    left = clamp(x - pad_x, 0, width - 1)
    top = clamp(y - pad_y, 0, height - 1)
    right = clamp(x + w + pad_x, left + 1, width)
    bottom = clamp(y + h + pad_y, top + 1, height)
    return left, top, right, bottom


def build_model(torch, models, checkpoint):
    """Recreate the trained architecture and load checkpoint weights."""
    model_name = checkpoint["model_name"]
    classes = checkpoint["classes"]
    num_classes = len(classes)

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model_name in checkpoint: {model_name}")

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, classes


def make_transform(transforms, image_size):
    """Create validation preprocessing for crop classification."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def evaluate_one(image_path, json_path, model, classes, transform, device, torch, padding, helmet_threshold):
    """Evaluate all labelled faces in one saved frame/json pair."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    faces = data.get("faces", [])
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")

    results = []
    with torch.no_grad():
        for face in faces:
            crop = image.crop(padded_bbox(face["bbox"], image.size, padding))
            tensor = transform(crop).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu()
            prob_by_class = {classes[i]: float(probs[i]) for i in range(len(classes))}
            helmet_prob = prob_by_class.get("helmet", 0.0)
            predicted = "helmet" if helmet_prob >= helmet_threshold else "no_helmet"
            results.append(
                {
                    "face_id": face.get("face_id"),
                    "name": face.get("name", ""),
                    "bbox": face["bbox"],
                    "expected": "helmet" if face.get("has_helmet") else "no_helmet",
                    "predicted": predicted,
                    "helmet_probability": helmet_prob,
                    "probabilities": prob_by_class,
                }
            )

    if len(faces) == 0:
        status = "LOCKED_NO_PERSON"
    elif len(faces) > 1:
        status = "LOCKED_MULTIPLE_PEOPLE"
    elif results[0]["predicted"] != "helmet":
        status = "LOCKED_NO_HELMET"
    else:
        status = "UNLOCK_CANDIDATE_HELMET_OK"

    return {
        "image": str(image_path),
        "json": str(json_path),
        "face_count": len(faces),
        "status": status,
        "faces": results,
    }


def iter_pairs(input_dir):
    """Yield image/json pairs from a scooter capture directory."""
    input_dir = Path(input_dir)
    for json_path in sorted(input_dir.glob("*.json")):
        if json_path.name.startswith("_"):
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        image_path = input_dir / data["filename"]
        if image_path.exists():
            yield image_path, json_path


def main():
    """Run PyTorch helmet evaluation on one image or a folder of captures."""
    parser = argparse.ArgumentParser(description="Evaluate helmet classifier on full frames using scooter JSON bbox format.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image")
    parser.add_argument("--json")
    parser.add_argument("--input-dir")
    parser.add_argument("--padding", type=float, default=0.25)
    parser.add_argument("--helmet-threshold", type=float, default=0.5)
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    if args.input_dir is None and (args.image is None or args.json is None):
        raise SystemExit("Pass either --input-dir or both --image and --json.")

    torch, models, transforms = require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model, classes = build_model(torch, models, checkpoint)
    model.to(device)
    transform = make_transform(transforms, checkpoint.get("image_size", 224))

    if args.input_dir:
        pairs = list(iter_pairs(args.input_dir))
    else:
        pairs = [(Path(args.image), Path(args.json))]

    outputs = [
        evaluate_one(image_path, json_path, model, classes, transform, device, torch, args.padding, args.helmet_threshold)
        for image_path, json_path in pairs
    ]

    for output in outputs:
        print(f"{output['status']} | faces={output['face_count']} | {Path(output['image']).name}")
        for face in output["faces"]:
            print(
                f"  face {face['face_id']}: expected={face['expected']} "
                f"predicted={face['predicted']} helmet_prob={face['helmet_probability']:.3f} "
                f"name={face['name']}"
            )

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(outputs, indent=2), encoding="utf-8")

    if args.output_csv:
        rows = []
        for output in outputs:
            if output["faces"]:
                for face in output["faces"]:
                    rows.append(
                        {
                            "image": Path(output["image"]).name,
                            "face_count": output["face_count"],
                            "status": output["status"],
                            "face_id": face["face_id"],
                            "name": face["name"],
                            "expected": face["expected"],
                            "predicted": face["predicted"],
                            "helmet_probability": f"{face['helmet_probability']:.6f}",
                        }
                    )
            else:
                rows.append(
                    {
                        "image": Path(output["image"]).name,
                        "face_count": output["face_count"],
                        "status": output["status"],
                        "face_id": "",
                        "name": "",
                        "expected": "",
                        "predicted": "",
                        "helmet_probability": "",
                    }
                )
        fieldnames = [
            "image",
            "face_count",
            "status",
            "face_id",
            "name",
            "expected",
            "predicted",
            "helmet_probability",
        ]
        with Path(args.output_csv).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
