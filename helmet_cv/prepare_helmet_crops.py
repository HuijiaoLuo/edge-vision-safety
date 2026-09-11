import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


# The released pipeline matches the final Arduino model: clean measured crops
# plus standard geometric/photometric augmentation, with no helmet recoloring.
LABEL_NAMES = {False: "no_helmet", True: "helmet"}


def clamp(value, low, high):
    """Clamp a bounding-box coordinate to the image limits."""
    return max(low, min(high, value))


def padded_bbox(bbox, image_size, padding):
    """Expand a face/head bbox before cropping the helmet ROI."""
    width, height = image_size
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    pad_x = int(round(w * padding))
    pad_y = int(round(h * padding))
    left = clamp(x - pad_x, 0, width - 1)
    top = clamp(y - pad_y, 0, height - 1)
    right = clamp(x + w + pad_x, left + 1, width)
    bottom = clamp(y + h + pad_y, top + 1, height)
    return left, top, right, bottom


def collect_faces(input_dir):
    """Read scooter-app JSON files and return one training sample per face box."""
    input_dir = Path(input_dir)
    samples = []
    for json_path in sorted(input_dir.glob("*.json")):
        if json_path.name.startswith("_"):
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        image_path = input_dir / data["filename"]
        if not image_path.exists():
            print(f"Skipping {json_path.name}: missing {image_path.name}")
            continue
        for face in data.get("faces", []):
            samples.append(
                {
                    "json": json_path,
                    "image": image_path,
                    "image_name": image_path.name,
                    "face_id": face["face_id"],
                    "name": face.get("name", ""),
                    "label": bool(face["has_helmet"]),
                    "bbox": face["bbox"],
                }
            )
    return samples


def split_by_image(samples, val_fraction, seed):
    """Split by source image so faces from one photo do not leak across sets."""
    rng = random.Random(seed)
    image_names = sorted({sample["image_name"] for sample in samples})

    # Keep the split image-level so faces from the same photo do not leak. The
    # dataset is tiny, so choose validation images greedily to preserve labels.
    per_image_counts = {}
    total_counts = {False: 0, True: 0}
    for image_name in image_names:
        counts = {False: 0, True: 0}
        for sample in samples:
            if sample["image_name"] == image_name:
                counts[sample["label"]] += 1
                total_counts[sample["label"]] += 1
        per_image_counts[image_name] = counts

    target_counts = {
        label: max(1, round(count * val_fraction))
        for label, count in total_counts.items()
        if count > 0
    }
    val_images = set()
    val_counts = {False: 0, True: 0}
    candidates = list(image_names)
    rng.shuffle(candidates)
    while True:
        deficits = {
            label: target_counts[label] - val_counts[label]
            for label in target_counts
        }
        if all(deficit <= 0 for deficit in deficits.values()):
            break
        best_image = None
        best_gain = -1
        for image_name in candidates:
            if image_name in val_images:
                continue
            gain = sum(
                min(max(0, deficits[label]), per_image_counts[image_name][label])
                for label in target_counts
            )
            if gain > best_gain:
                best_gain = gain
                best_image = image_name
        if best_image is None or best_gain <= 0:
            break
        val_images.add(best_image)
        for label in val_counts:
            val_counts[label] += per_image_counts[best_image][label]

    split = {}
    for sample in samples:
        split[sample["image_name"]] = "val" if sample["image_name"] in val_images else "train"
    return split


def augment_crop(image, rng):
    """Apply conservative geometric/photometric augmentation to training crops."""
    img = image.copy()
    if rng.random() < 0.5:
        img = ImageOps.mirror(img)
    if rng.random() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.25))
    if rng.random() < 0.8:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.75, 1.35))
    if rng.random() < 0.6:
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.75, 1.25))
    if rng.random() < 0.5:
        img = img.rotate(rng.uniform(-12, 12), resample=Image.Resampling.BILINEAR, expand=False)
    return img


def save_crop(crop, output_path, image_size):
    """Resize/pad a crop to the classifier input size and save it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop = ImageOps.exif_transpose(crop).convert("RGB")
    crop = ImageOps.pad(crop, (image_size, image_size), method=Image.Resampling.BICUBIC)
    crop.save(output_path, quality=94)


def main():
    """Create an ImageFolder-style helmet/no-helmet crop dataset."""
    parser = argparse.ArgumentParser(description="Prepare helmet/no-helmet face crops from TRAIN_READY JSON labels.")
    parser.add_argument("--input-dir", nargs="+", default=["TRAIN_READY"])
    parser.add_argument("--output-dir", default="helmet_cv/dataset")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--padding", type=float, default=0.25)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--augment-per-train-crop", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dirs = [Path(input_dir) for input_dir in args.input_dir]
    output_dir = Path(args.output_dir)
    rng = random.Random(args.seed)

    samples = []
    for input_dir in input_dirs:
        samples.extend(collect_faces(input_dir))
    if not samples:
        raise SystemExit(f"No labelled faces found in {input_dirs}")

    split_by_name = split_by_image(samples, args.val_fraction, args.seed)
    rows = []

    for sample in samples:
        split = split_by_name[sample["image_name"]]
        label_name = LABEL_NAMES[sample["label"]]
        with Image.open(sample["image"]) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            crop_box = padded_bbox(sample["bbox"], img.size, args.padding)
            crop = img.crop(crop_box)

        stem = f"{Path(sample['image_name']).stem}_face{sample['face_id']}"
        original_path = output_dir / split / label_name / f"{stem}.jpg"
        save_crop(crop, original_path, args.image_size)
        rows.append(
            {
                "split": split,
                "path": str(original_path),
                "label": label_name,
                "source_image": sample["image_name"],
                "face_id": sample["face_id"],
                "person_name": sample["name"],
                "augmented": False,
                "bbox_x": sample["bbox"]["x"],
                "bbox_y": sample["bbox"]["y"],
                "bbox_w": sample["bbox"]["w"],
                "bbox_h": sample["bbox"]["h"],
            }
        )

        if split == "train":
            for idx in range(args.augment_per_train_crop):
                aug = augment_crop(crop, rng)
                aug_path = output_dir / split / label_name / f"{stem}_aug{idx:02d}.jpg"
                save_crop(aug, aug_path, args.image_size)
                aug_row = dict(rows[-1])
                aug_row["path"] = str(aug_path)
                aug_row["augmented"] = True
                rows.append(aug_row)

    metadata_path = output_dir / "metadata.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for row in rows:
        counts[(row["split"], row["label"])] = counts.get((row["split"], row["label"]), 0) + 1
    print(f"Prepared {len(rows)} crops at {output_dir}")
    for key in sorted(counts):
        print(f"{key[0]:5s} {key[1]:9s}: {counts[key]}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
