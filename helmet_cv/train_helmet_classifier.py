import argparse
import copy
import json
import random
from pathlib import Path


def require_torch():
    """Import PyTorch dependencies only when training is requested."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torchvision import datasets, models, transforms
    except ImportError as exc:
        raise SystemExit(
            "Missing PyTorch/torchvision. Install with:\n"
            "  python -m pip install torch torchvision scikit-learn\n"
            "Then run this script again."
        ) from exc
    return torch, nn, DataLoader, datasets, models, transforms


def set_seed(seed):
    """Make data loading and augmentation reasonably reproducible."""
    random.seed(seed)
    torch, *_ = require_torch()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(transforms, image_size):
    """Create train/validation transforms for ImageFolder crops."""
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.03),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tfms, eval_tfms


def build_model(models, nn, model_name, num_classes, freeze_backbone):
    """Create the selected torchvision classifier and replace its head."""
    if model_name == "mobilenet_v3_small":
        try:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            model = models.mobilenet_v3_small(weights=weights)
        except Exception:
            model = models.mobilenet_v3_small(weights=None)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        try:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            model = models.efficientnet_b0(weights=weights)
        except Exception:
            model = models.efficientnet_b0(weights=None)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def class_weights(dataset, torch, device):
    """Compute inverse-frequency class weights for imbalanced crops."""
    counts = [0 for _ in dataset.classes]
    for _, label in dataset.samples:
        counts[label] += 1
    total = sum(counts)
    weights = [total / max(1, len(counts) * count) for count in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device), counts


def train_one_epoch(model, loader, criterion, optimizer, device, torch):
    """Run one supervised training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / max(1, total), correct / max(1, total)


def evaluate(model, loader, criterion, device, torch):
    """Evaluate loss, accuracy, and confusion matrix on validation data."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    confusion = None
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            running_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if confusion is None:
                n = logits.shape[1]
                confusion = torch.zeros((n, n), dtype=torch.int64)
            for true_label, pred_label in zip(labels.cpu(), preds.cpu()):
                confusion[true_label, pred_label] += 1

    return running_loss / max(1, total), correct / max(1, total), confusion


def main():
    """Train and save a helmet/no-helmet image classifier."""
    parser = argparse.ArgumentParser(description="Fine-tune a compact helmet classifier from cropped faces.")
    parser.add_argument("--data-dir", default="helmet_cv/dataset")
    parser.add_argument("--output-dir", default="helmet_cv/runs/mobilenet_v3_small")
    parser.add_argument("--model", choices=["mobilenet_v3_small", "efficientnet_b0"], default="mobilenet_v3_small")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch, nn, DataLoader, datasets, models, transforms = require_torch()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_tfms, eval_tfms = build_transforms(transforms, args.image_size)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(models, nn, args.model, len(train_ds.classes), args.freeze_backbone).to(device)
    weights, counts = class_weights(train_ds, torch, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_acc = -1.0
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    history = []
    print(f"Device: {device}")
    print(f"Classes: {train_ds.classes}")
    print(f"Train class counts: {dict(zip(train_ds.classes, counts))}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, torch)
        val_loss, val_acc, confusion = evaluate(model, val_loader, criterion, device, torch)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "confusion": confusion.tolist() if confusion is not None else None,
            }
        )
        print(
            f"epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc = val_acc
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = {
        "model_name": args.model,
        "state_dict": model.state_dict(),
        "classes": train_ds.classes,
        "image_size": args.image_size,
        "history": history,
        "freeze_backbone": args.freeze_backbone,
    }
    torch.save(checkpoint, output_dir / "helmet_classifier.pt")

    model.eval()
    example = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    traced = torch.jit.trace(model, example)
    traced.save(str(output_dir / "helmet_classifier_torchscript.pt"))

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Best epoch: {best_epoch}")
    print(f"Best val accuracy: {best_acc:.3f}")
    print(f"Best val loss: {best_loss:.4f}")
    print(f"Saved checkpoint: {output_dir / 'helmet_classifier.pt'}")
    print(f"Saved TorchScript: {output_dir / 'helmet_classifier_torchscript.pt'}")


if __name__ == "__main__":
    main()
