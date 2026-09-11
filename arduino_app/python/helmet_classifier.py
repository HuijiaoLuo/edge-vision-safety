import logging
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_CANDIDATES = [
    BASE_DIR.parent / "models" / "helmet_classifier.onnx",
    BASE_DIR / "models" / "helmet_classifier.onnx",
]

IMAGE_SIZE = 224
PADDING = 0.25
HELMET_THRESHOLD = 0.5
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _clamp(value: int, low: int, high: int) -> int:
    """Clamp a coordinate to the valid image range."""
    return max(low, min(high, value))


def padded_bbox(face, image_shape, padding: float = PADDING):
    """Expand the detected face box into the head/helmet ROI used by the classifier."""
    height, width = image_shape[:2]
    x, y, w, h = face
    pad_x = int(round(w * padding))
    pad_y = int(round(h * padding))
    left = _clamp(int(x - pad_x), 0, width - 1)
    top = _clamp(int(y - pad_y), 0, height - 1)
    right = _clamp(int(x + w + pad_x), left + 1, width)
    bottom = _clamp(int(y + h + pad_y), top + 1, height)
    return left, top, right, bottom


def preprocess_bgr_crop(crop_bgr: np.ndarray, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Match the ImageNet preprocessing used during EfficientNet training."""
    crop = cv2.resize(crop_bgr, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    crop = (crop - IMAGENET_MEAN) / IMAGENET_STD
    crop = np.transpose(crop, (2, 0, 1))
    return crop[np.newaxis, :, :, :].astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Convert raw classifier logits into class probabilities."""
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


class HelmetClassifier:
    """OpenCV-DNN wrapper for the exported helmet/no-helmet ONNX classifier."""

    def __init__(
            self,
            model_path: str | Path | None = None,
            threshold: float = HELMET_THRESHOLD,
            padding: float = PADDING,
            logger: logging.Logger | None = None,
    ):
        """Load the ONNX model and keep runtime threshold/padding settings."""
        self.logger = logger or logging.getLogger(__name__)
        self.model_path = Path(model_path) if model_path else self._find_default_model()
        self.threshold = threshold
        self.padding = padding
        self.net = None

        if self.model_path.exists():
            self.logger.info(f"[HelmetClassifier] Loading ONNX model: {self.model_path}")
            self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
            self.logger.info("[HelmetClassifier] Model loaded")
        else:
            self.logger.warning(f"[HelmetClassifier] Model not found: {self.model_path}")

    @staticmethod
    def _find_default_model() -> Path:
        """Find the model in either Arduino `/app` layout or local repo layout."""
        for candidate in DEFAULT_MODEL_CANDIDATES:
            if candidate.exists():
                return candidate
        return DEFAULT_MODEL_CANDIDATES[0]

    def is_available(self) -> bool:
        """Return whether the helmet ONNX model was loaded successfully."""
        return self.net is not None

    def predict_crop(self, crop_bgr: np.ndarray) -> dict:
        """Run helmet/no-helmet classification on one cropped ROI."""
        if self.net is None:
            return {
                "predicted": "unknown",
                "helmet_probability": None,
                "no_helmet_probability": None,
                "available": False,
            }

        blob = preprocess_bgr_crop(crop_bgr)
        self.net.setInput(blob)
        logits = self.net.forward()
        probs = softmax(logits)[0]

        helmet_probability = float(probs[0])
        no_helmet_probability = float(probs[1]) if len(probs) > 1 else 1.0 - helmet_probability
        predicted = "helmet" if helmet_probability >= self.threshold else "no_helmet"

        return {
            "predicted": predicted,
            "helmet_probability": helmet_probability,
            "no_helmet_probability": no_helmet_probability,
            "available": True,
        }

    def predict_faces(self, frame_bgr: np.ndarray | None, faces: list) -> list[dict]:
        """Classify each detected face ROI and keep one result per face."""
        if frame_bgr is None:
            return []

        predictions = []
        for idx, face in enumerate(faces):
            left, top, right, bottom = padded_bbox(face, frame_bgr.shape, self.padding)
            crop = frame_bgr[top:bottom, left:right]
            result = self.predict_crop(crop)
            result.update({
                "face_id": idx + 1,
                "bbox": {
                    "x": int(face[0]),
                    "y": int(face[1]),
                    "w": int(face[2]),
                    "h": int(face[3]),
                },
            })
            predictions.append(result)
        return predictions


def safety_status(face_count: int, helmet_predictions: list[dict]) -> str:
    """Helmet-only status helper kept for standalone classifier checks."""
    if face_count == 0:
        return "LOCKED_NO_PERSON"
    if face_count > 1:
        return "LOCKED_MULTIPLE_PEOPLE"
    if not helmet_predictions:
        return "LOCKED_NO_HELMET_MODEL"
    if helmet_predictions[0].get("predicted") == "helmet":
        return "UNLOCK_CANDIDATE_HELMET_OK"
    return "LOCKED_NO_HELMET"
