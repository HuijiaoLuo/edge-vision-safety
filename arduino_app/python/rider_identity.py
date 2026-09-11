import json
import logging
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_CANDIDATES = [
    Path("/app/models/face_recognition.onnx"),
    BASE_DIR.parent / "models" / "face_recognition.onnx",
    BASE_DIR / "models" / "face_recognition.onnx",
]
DEFAULT_DETECTOR_CANDIDATES = [
    Path("/app/models/face_detection_yunet.onnx"),
    BASE_DIR.parent / "models" / "face_detection_yunet.onnx",
    BASE_DIR / "models" / "face_detection_yunet.onnx",
]
DEFAULT_USERS_DIR_CANDIDATES = [
    Path("/app/users"),
    BASE_DIR.parent / "users",
    BASE_DIR / "users",
]

FACE_IMAGE_SIZE = 112
COSINE_THRESHOLD = 0.45


def _find_first_existing(candidates: list[Path]) -> Path:
    """Return the first existing path, or the first candidate as a fallback."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_app_path(path: str | Path | None, fallback_candidates: list[Path]) -> Path:
    """Resolve paths both in SSH layout and Arduino App Lab's `/app` runtime."""
    if path is None:
        return _find_first_existing(fallback_candidates)

    candidate = Path(path)
    if candidate.exists():
        return candidate

    text = str(candidate)
    host_prefix = "/home/arduino/ArduinoApps/who-wears-the-helmet"
    if text.startswith(host_prefix):
        app_candidate = Path("/app") / text[len(host_prefix):].lstrip("/")
        if app_candidate.exists():
            return app_candidate

    for fallback in fallback_candidates:
        if fallback.exists():
            return fallback
    return candidate


def _clamp(value: int, low: int, high: int) -> int:
    """Clamp a coordinate to the valid image range."""
    return max(low, min(high, value))


def crop_face(frame_bgr: np.ndarray, face, padding: float = 0.15) -> np.ndarray | None:
    """Fallback crop when landmark-based alignment is unavailable."""
    height, width = frame_bgr.shape[:2]
    x, y, w, h = face
    pad_x = int(round(w * padding))
    pad_y = int(round(h * padding))
    left = _clamp(int(x - pad_x), 0, width - 1)
    top = _clamp(int(y - pad_y), 0, height - 1)
    right = _clamp(int(x + w + pad_x), left + 1, width)
    bottom = _clamp(int(y + h + pad_y), top + 1, height)
    crop = frame_bgr[top:bottom, left:right]
    if crop.size == 0:
        return None
    return crop


def preprocess_face(crop_bgr: np.ndarray) -> np.ndarray:
    """Prepare a face crop for direct ONNX inference fallback."""
    face = cv2.resize(crop_bgr, (FACE_IMAGE_SIZE, FACE_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
    face = (face - 127.5) / 128.0
    face = np.transpose(face, (2, 0, 1))
    return face[np.newaxis, :, :, :].astype(np.float32)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Normalize embeddings so cosine distance is stable."""
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Smaller distance means the live face is closer to a registered embedding."""
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(1.0 - np.dot(a, b))


def load_embedding(path: Path) -> np.ndarray:
    """Load a registered rider reference embedding from `.npy` or JSON."""
    if path.suffix.lower() == ".npy":
        return l2_normalize(np.load(path))
    with open(path, encoding="utf-8") as f:
        return l2_normalize(np.asarray(json.load(f), dtype=np.float32))


def save_embedding(path: Path, embedding: np.ndarray) -> None:
    """Save a normalized registered rider reference embedding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npy":
        np.save(path, l2_normalize(embedding))
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(l2_normalize(embedding).tolist(), f)


class RiderRecognizer:
    """Register and recognize riders with OpenCV YuNet + SFace embeddings."""

    def __init__(
            self,
            model_path: str | Path | None = None,
            detector_path: str | Path | None = None,
            users_dir: str | Path | None = None,
            threshold: float = COSINE_THRESHOLD,
            logger: logging.Logger | None = None,
    ):
        """Load face-recognition models and registered rider references."""
        self.logger = logger or logging.getLogger(__name__)
        self.model_path = _resolve_app_path(model_path, DEFAULT_MODEL_CANDIDATES)
        self.detector_path = _resolve_app_path(detector_path, DEFAULT_DETECTOR_CANDIDATES)
        self.users_dir = _resolve_app_path(users_dir, DEFAULT_USERS_DIR_CANDIDATES)
        self.threshold = threshold
        self.net = None
        self.sf_model = None
        self.detector = None
        self.references: dict[str, np.ndarray] = {}

        if self.model_path.exists():
            self.logger.info(f"[RiderRecognizer] Loading face recognition ONNX model: {self.model_path}")
            if hasattr(cv2, "FaceRecognizerSF_create"):
                self.sf_model = cv2.FaceRecognizerSF_create(str(self.model_path), "")
            elif hasattr(cv2, "FaceRecognizerSF") and hasattr(cv2.FaceRecognizerSF, "create"):
                self.sf_model = cv2.FaceRecognizerSF.create(str(self.model_path), "")
            else:
                self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
            self.logger.info("[RiderRecognizer] Model loaded")
        else:
            self.logger.warning(f"[RiderRecognizer] Face recognition model not found: {self.model_path}")

        if self.detector_path.exists() and hasattr(cv2, "FaceDetectorYN_create"):
            self.logger.info(f"[RiderRecognizer] Loading YuNet detector ONNX model: {self.detector_path}")
            self.detector = cv2.FaceDetectorYN_create(str(self.detector_path), "", (320, 320))
            self.logger.info("[RiderRecognizer] YuNet detector loaded")
        elif self.detector_path.exists() and hasattr(cv2, "FaceDetectorYN") and hasattr(cv2.FaceDetectorYN, "create"):
            self.logger.info(f"[RiderRecognizer] Loading YuNet detector ONNX model: {self.detector_path}")
            self.detector = cv2.FaceDetectorYN.create(str(self.detector_path), "", (320, 320))
            self.logger.info("[RiderRecognizer] YuNet detector loaded")
        else:
            self.logger.warning(f"[RiderRecognizer] YuNet detector unavailable or not found: {self.detector_path}")

        self.reload_users()

    def is_available(self) -> bool:
        """Return whether an embedding model is ready for inference."""
        return self.sf_model is not None or self.net is not None

    def reload_users(self) -> None:
        """Reload all registered rider embeddings from the users directory."""
        self.references = {}
        if not self.users_dir.exists():
            self.logger.warning(f"[RiderRecognizer] Users directory not found: {self.users_dir}")
            return

        for path in sorted(self.users_dir.iterdir()):
            if path.suffix.lower() not in {".npy", ".json"}:
                continue
            try:
                self.references[path.stem] = load_embedding(path)
            except Exception as exc:
                self.logger.warning(f"[RiderRecognizer] Failed to load {path}: {exc}")

        self.logger.info(f"[RiderRecognizer] Registered users loaded: {sorted(self.references)}")

    def embedding_from_crop(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Convert one face crop into a normalized identity embedding."""
        if not self.is_available():
            return None

        face = cv2.resize(crop_bgr, (FACE_IMAGE_SIZE, FACE_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        if self.sf_model is not None:
            embedding = self.sf_model.feature(face)
        else:
            blob = preprocess_face(face)
            self.net.setInput(blob)
            embedding = self.net.forward()
        return l2_normalize(embedding)

    def _aligned_crop_from_frame(self, frame_bgr: np.ndarray, target_face) -> np.ndarray | None:
        """Find landmarks near the target face box and align the face for SFace."""
        if self.detector is None or self.sf_model is None:
            return None

        height, width = frame_bgr.shape[:2]
        self.detector.setInputSize((width, height))
        _, detections = self.detector.detect(frame_bgr)
        if detections is None or len(detections) == 0:
            return None

        tx, ty, tw, th = [float(v) for v in target_face]
        tcx = tx + tw / 2.0
        tcy = ty + th / 2.0

        best = None
        best_score = None
        for detection in detections:
            x, y, w, h = [float(v) for v in detection[:4]]
            cx = x + w / 2.0
            cy = y + h / 2.0
            center_dist = ((cx - tcx) ** 2.0 + (cy - tcy) ** 2.0) ** 0.5
            size = max(tw, th, 1.0)
            score = center_dist / size
            if best_score is None or score < best_score:
                best = detection
                best_score = score

        if best is None or best_score is None or best_score > 0.65:
            return None
        try:
            return self.sf_model.alignCrop(frame_bgr, best)
        except Exception as exc:
            self.logger.debug(f"[RiderRecognizer] alignCrop failed: {exc}")
            return None

    def embedding_from_face(self, frame_bgr: np.ndarray | None, face) -> np.ndarray | None:
        """Create an embedding for one detected face box in a full frame."""
        if frame_bgr is None:
            return None
        aligned = self._aligned_crop_from_frame(frame_bgr, face)
        if aligned is not None:
            return self.embedding_from_crop(aligned)
        crop = crop_face(frame_bgr, face)
        if crop is None:
            return None
        return self.embedding_from_crop(crop)

    def recognize_face(self, frame_bgr: np.ndarray | None, face, face_id: int = 1) -> dict:
        """Compare one live face embedding against registered users."""
        base = {
            "face_id": face_id,
            "authorized": False,
            "registration_status": "UNREGISTERED",
            "user": None,
            "display_user": "UNREGISTERED",
            "cosine_distance": None,
            "available": self.is_available(),
            "registered_users": sorted(self.references),
        }

        if not self.is_available():
            base["status"] = "NO_USER_MODEL"
            base["display_user"] = "NO_USER_MODEL"
            return base
        if not self.references:
            base["status"] = "NO_REGISTERED_USERS"
            base["display_user"] = "NO_REGISTERED_USERS"
            return base

        embedding = self.embedding_from_face(frame_bgr, face)
        if embedding is None:
            base["status"] = "NO_FACE_EMBEDDING"
            return base

        best_user = None
        best_distance = None
        for user, reference in self.references.items():
            distance = cosine_distance(embedding, reference)
            if best_distance is None or distance < best_distance:
                best_user = user
                best_distance = distance

        authorized = best_distance is not None and best_distance <= self.threshold
        base.update({
            "authorized": bool(authorized),
            "registration_status": "REGISTERED" if authorized else "UNREGISTERED",
            "user": best_user if authorized else None,
            "display_user": best_user if authorized else "UNREGISTERED",
            "best_match": best_user,
            "cosine_distance": best_distance,
            "status": "AUTHORIZED" if authorized else "UNKNOWN_RIDER",
        })
        return base

    def recognize_faces(self, frame_bgr: np.ndarray | None, faces: list) -> list[dict]:
        """Recognize every detected face and preserve face order."""
        return [
            self.recognize_face(frame_bgr, face, idx + 1)
            for idx, face in enumerate(faces)
        ]


def final_scooter_status(
        face_count: int,
        helmet_predictions: list[dict],
        rider_predictions: list[dict],
        require_registered_user: bool = True,
) -> str:
    """Final safety gate: exactly one authorized rider wearing a helmet unlocks."""
    if face_count == 0:
        return "LOCKED_NO_PERSON"
    if face_count > 1:
        return "LOCKED_MULTIPLE_PEOPLE"

    if require_registered_user:
        if not rider_predictions:
            return "LOCKED_NO_USER_RECOGNITION"
        rider = rider_predictions[0]
        if rider.get("status") == "NO_USER_MODEL":
            return "LOCKED_NO_USER_MODEL"
        if rider.get("status") == "NO_REGISTERED_USERS":
            return "LOCKED_NO_REGISTERED_USERS"
        if not rider.get("authorized", False):
            return "LOCKED_UNAUTHORIZED_RIDER"

    if not helmet_predictions:
        return "LOCKED_NO_HELMET_MODEL"
    if helmet_predictions[0].get("predicted") != "helmet":
        return "LOCKED_NO_HELMET"

    return "UNLOCKED" if require_registered_user else "UNLOCK_CANDIDATE_HELMET_OK"
