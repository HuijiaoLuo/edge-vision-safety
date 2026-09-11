"""Starter-free safety logic for the scooter helmet prototype.

The hackathon Arduino App Lab camera loop supplied `frame_bgr` and detected
face boxes, then called this module to produce the final lock status.
"""

from helmet_classifier import HelmetClassifier
from rider_identity import RiderRecognizer, final_scooter_status


def evaluate_scooter_frame(
        frame_bgr,
        faces,
        helmet_classifier: HelmetClassifier,
        rider_recognizer: RiderRecognizer,
        require_registered_user: bool = True,
) -> dict:
    """Evaluate one frame and return the final status plus per-face outputs."""
    helmet_predictions = helmet_classifier.predict_faces(frame_bgr, faces)
    rider_predictions = rider_recognizer.recognize_faces(frame_bgr, faces)
    status = final_scooter_status(
        len(faces),
        helmet_predictions,
        rider_predictions,
        require_registered_user=require_registered_user,
    )
    return {
        "face_count": len(faces),
        "status": status,
        "helmet_predictions": helmet_predictions,
        "rider_predictions": rider_predictions,
    }


def compact_log_fields(result: dict) -> dict:
    """Format model outputs for live logs or UI status payloads."""
    helmet_labels = [pred.get("predicted") for pred in result.get("helmet_predictions", [])]
    helmet_probs = [
        None if pred.get("helmet_probability") is None else round(pred["helmet_probability"], 3)
        for pred in result.get("helmet_predictions", [])
    ]
    riders = []
    for pred in result.get("rider_predictions", []):
        if pred.get("authorized"):
            riders.append(f"REGISTERED:{pred.get('user')}")
        elif pred.get("best_match") is not None:
            riders.append(f"UNREGISTERED(best={pred.get('best_match')},dist={pred.get('cosine_distance'):.3f})")
        else:
            riders.append(pred.get("display_user") or pred.get("status"))

    return {
        "faces": result.get("face_count", 0),
        "status": result.get("status"),
        "helmet_labels": helmet_labels,
        "helmet_probs": helmet_probs,
        "riders": riders,
    }
