import argparse
import json
from pathlib import Path

import cv2

from helmet_classifier import HelmetClassifier
from rider_identity import RiderRecognizer, final_scooter_status


def load_faces(json_path: Path) -> list[tuple[int, int, int, int]]:
    """Read face boxes from one saved scooter-app annotation JSON."""
    with open(json_path, encoding="utf-8") as f:
        metadata = json.load(f)
    faces = []
    for face in metadata.get("faces", []):
        bbox = face.get("bbox")
        if not bbox:
            continue
        faces.append((int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"])))
    return faces


def main():
    """Evaluate rider registration and helmet status for one saved frame."""
    parser = argparse.ArgumentParser(description="Evaluate registered rider and helmet status on one saved frame.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--helmet-model", default="../models/helmet_classifier.onnx")
    parser.add_argument("--face-model", default="../models/face_recognition.onnx")
    parser.add_argument("--face-detector", default="../models/face_detection_yunet.onnx")
    parser.add_argument("--users-dir", default="../users")
    parser.add_argument("--rider-threshold", type=float, default=0.45)
    parser.add_argument("--helmet-threshold", type=float, default=0.5)
    args = parser.parse_args()

    frame = cv2.imread(str(Path(args.image)))
    if frame is None:
        raise SystemExit(f"Could not read image: {args.image}")

    faces = load_faces(Path(args.json))
    helmet = HelmetClassifier(model_path=args.helmet_model, threshold=args.helmet_threshold)
    rider = RiderRecognizer(
        model_path=args.face_model,
        detector_path=args.face_detector,
        users_dir=args.users_dir,
        threshold=args.rider_threshold,
    )

    helmet_predictions = helmet.predict_faces(frame, faces)
    rider_predictions = rider.recognize_faces(frame, faces)
    status = final_scooter_status(
        len(faces),
        helmet_predictions,
        rider_predictions,
        require_registered_user=True,
    )

    print(f"faces={len(faces)} status={status}")
    for idx, (helmet_pred, rider_pred) in enumerate(zip(helmet_predictions, rider_predictions), start=1):
        if rider_pred.get("authorized"):
            rider_text = f"REGISTERED:{rider_pred.get('user')}"
        else:
            rider_text = "UNREGISTERED"
        print(
            f"face={idx} "
            f"rider={rider_text} "
            f"best_match={rider_pred.get('best_match')} "
            f"rider_distance={rider_pred.get('cosine_distance')} "
            f"helmet={helmet_pred.get('predicted')} "
            f"helmet_prob={helmet_pred.get('helmet_probability')}"
        )


if __name__ == "__main__":
    main()
