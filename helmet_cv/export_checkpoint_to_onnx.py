import argparse
from pathlib import Path


def require_torch():
    """Import PyTorch only when exporting a checkpoint."""
    try:
        import torch
        from torchvision import models
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/torchvision. Run this on the training machine, not on Arduino:\n"
            "  python -m pip install -r helmet_cv/requirements.txt"
        ) from exc
    return torch, models


def build_model(torch, models, checkpoint):
    """Rebuild the saved classifier architecture for ONNX export."""
    model_name = checkpoint["model_name"]
    num_classes = len(checkpoint["classes"])
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def main():
    """Export a trained PyTorch helmet checkpoint to ONNX."""
    parser = argparse.ArgumentParser(description="Export a trained helmet classifier checkpoint to ONNX.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch, models = require_torch()
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    model = build_model(torch, models, checkpoint)
    image_size = checkpoint.get("image_size", 224)
    dummy = torch.randn(1, 3, image_size, image_size)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX model: {output}")
    print(f"Classes: {checkpoint['classes']}")
    print(f"Image size: {image_size}")


if __name__ == "__main__":
    main()
