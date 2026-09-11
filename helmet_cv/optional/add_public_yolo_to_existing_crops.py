import argparse
import csv
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


# Optional experiment: add public bike/scooter helmet crops to the training set
# while keeping the local measured validation split unchanged. This was useful
# as a generalization check, but the final Arduino deployment used the clean
# measured-data model.
CLASS_TO_FOLDER = {
    0: "helmet",
    1: "no_helmet",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def clamp(value, low, high):
    """Clamp a coordinate to an image boundary."""
    return max(low, min(high, value))


def yolo_to_xywh(line, image_size):
    """Convert one YOLO label row into pixel-space xywh coordinates."""
    width, height = image_size
    class_id, cx, cy, bw, bh = line.split()[:5]
    class_id = int(float(class_id))
    cx = float(cx) * width
    cy = float(cy) * height
    w = float(bw) * width
    h = float(bh) * height
    x = cx - w / 2.0
    y = cy - h / 2.0
    return class_id, x, y, w, h


def padded_box(x, y, w, h, image_size, padding):
    """Expand a YOLO object box before cropping."""
    width, height = image_size
    pad_x = int(round(w * padding))
    pad_y = int(round(h * padding))
    left = clamp(int(round(x - pad_x)), 0, width - 1)
    top = clamp(int(round(y - pad_y)), 0, height - 1)
    right = clamp(int(round(x + w + pad_x)), left + 1, width)
    bottom = clamp(int(round(y + h + pad_y)), top + 1, height)
    return left, top, right, bottom


def augment_crop(image, rng):
    """Apply light augmentation to public training crops."""
    img = image.copy()
    if rng.random() < 0.5:
        img = ImageOps.mirror(img)
    if rng.random() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.25))
    if rng.random() < 0.8:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.75, 1.35))
    if rng.random() < 0.5:
        img = img.rotate(rng.uniform(-8, 8), resample=Image.Resampling.BILINEAR, expand=False)
    return img


def save_crop(crop, output_path, image_size):
    """Pad a crop to a square classifier input image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop = ImageOps.exif_transpose(crop).convert("RGB")
    crop = ImageOps.pad(crop, (image_size, image_size), method=Image.Resampling.BICUBIC)
    crop.save(output_path, quality=94)


def copy_existing_dataset(existing_dir, output_dir):
    """Copy the measured crop dataset before adding public crops."""
    existing_dir = Path(existing_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(existing_dir, output_dir)


def find_images(public_dir, split):
    """Find public YOLO images for one dataset split."""
    public_dir = Path(public_dir)
    folder = public_dir / split / "images"
    if not folder.exists():
        return []
    return [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]


def label_path_for_image(image_path):
    """Return the matching YOLO label path for an image path."""
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def collect_public_objects(public_dir, max_per_class, seed):
    """Sample helmet/no-helmet objects from a public YOLO dataset."""
    objects = []
    for split in ["train", "valid", "test"]:
        for image_path in find_images(public_dir, split):
            label_path = label_path_for_image(image_path)
            if not label_path.exists():
                continue
            with Image.open(image_path) as img:
                image_size = img.size
            for idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                class_id, x, y, w, h = yolo_to_xywh(line, image_size)
                if class_id not in CLASS_TO_FOLDER:
                    continue
                objects.append(
                    {
                        "image_path": image_path,
                        "source_split": split,
                        "object_index": idx,
                        "class_id": class_id,
                        "label": CLASS_TO_FOLDER[class_id],
                        "bbox": (x, y, w, h),
                    }
                )

    rng = random.Random(seed)
    selected = []
    for class_id in sorted(CLASS_TO_FOLDER):
        class_objects = [obj for obj in objects if obj["class_id"] == class_id]
        rng.shuffle(class_objects)
        selected.extend(class_objects[:max_per_class])
    return selected


def append_public_crops(objects, output_dir, image_size, padding, augment_per_crop, seed):
    """Append public object crops to the training split only."""
    rng = random.Random(seed)
    rows = []
    for obj in objects:
        with Image.open(obj["image_path"]) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            crop = img.crop(padded_box(*obj["bbox"], img.size, padding))

        stem = f"public_{obj['source_split']}_{obj['image_path'].stem}_obj{obj['object_index']}"
        label = obj["label"]
        out_path = Path(output_dir) / "train" / label / f"{stem}.jpg"
        save_crop(crop, out_path, image_size)
        rows.append(
            {
                "split": "train",
                "label": label,
                "path": str(out_path),
                "source_image": str(obj["image_path"]),
                "source_split": obj["source_split"],
                "augmented": False,
            }
        )

        for aug_idx in range(augment_per_crop):
            aug = augment_crop(crop, rng)
            aug_path = Path(output_dir) / "train" / label / f"{stem}_aug{aug_idx:02d}.jpg"
            save_crop(aug, aug_path, image_size)
            row = dict(rows[-1])
            row["path"] = str(aug_path)
            row["augmented"] = True
            rows.append(row)
    return rows


def count_dataset(dataset_dir):
    """Count images in each split/class folder."""
    counts = {}
    for split in ["train", "val"]:
        for label in ["helmet", "no_helmet"]:
            folder = Path(dataset_dir) / split / label
            counts[(split, label)] = len([p for p in folder.glob("*") if p.suffix.lower() in IMAGE_EXTS]) if folder.exists() else 0
    return counts


def main():
    """Build an optional local-plus-public crop dataset."""
    parser = argparse.ArgumentParser(description="Copy an existing crop dataset and append sampled public YOLO crops to train.")
    parser.add_argument("--existing-crop-dir", default="helmet_cv/dataset_clean")
    parser.add_argument("--public-yolo-dir", required=True)
    parser.add_argument("--output-dir", default="helmet_cv/dataset_public_local")
    parser.add_argument("--max-public-per-class", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--padding", type=float, default=0.25)
    parser.add_argument("--augment-public", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    copy_existing_dataset(args.existing_crop_dir, args.output_dir)
    objects = collect_public_objects(args.public_yolo_dir, args.max_public_per_class, args.seed)
    rows = append_public_crops(objects, args.output_dir, args.image_size, args.padding, args.augment_public, args.seed)

    metadata_path = Path(args.output_dir) / "public_added_metadata.csv"
    if rows:
        with metadata_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Copied existing crop dataset: {args.existing_crop_dir}")
    print(f"Added public crops: {len(rows)}")
    for key, value in sorted(count_dataset(args.output_dir).items()):
        print(f"{key[0]:5s} {key[1]:9s}: {value}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
