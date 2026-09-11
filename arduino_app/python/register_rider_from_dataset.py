import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from rider_identity import RiderRecognizer, save_embedding


def identity_from_json_name(name: str, mode: str) -> str | None:
    """Map labels like `Simon_front_hat` to the registered identity `simon`."""
    name = name.strip()
    if not name:
        return None
    if mode == "exact":
        return name
    first = re.split(r"[_\s]+", name)[0].strip()
    if not first:
        return None
    aliases = {
        "staphania": "stephania",
    }
    return aliases.get(first.lower(), first.lower())


def iter_labelled_faces(dataset_dir: Path, match_identity: str | None, identity_mode: str):
    """Yield every labelled face box from the scooter JSON files."""
    for json_path in sorted(dataset_dir.glob("*.json")):
        with open(json_path, encoding="utf-8") as f:
            metadata = json.load(f)

        image_path = dataset_dir / metadata.get("filename", "")
        if not image_path.exists():
            continue

        for face in metadata.get("faces", []):
            name = str(face.get("name", "")).strip()
            identity = identity_from_json_name(name, identity_mode)
            if identity is None:
                continue
            if match_identity and identity != match_identity:
                continue
            bbox = face.get("bbox")
            if not bbox:
                continue
            yield image_path, name, identity, (
                int(bbox["x"]),
                int(bbox["y"]),
                int(bbox["w"]),
                int(bbox["h"]),
            )


def main():
    """Build registered-user embeddings from labelled scooter-app captures."""
    parser = argparse.ArgumentParser(
        description="Register authorized riders by averaging face embeddings from JSON-labelled captures."
    )
    parser.add_argument("--dataset-dir", default="../dataset_helmet")
    parser.add_argument("--name", help="Register only this identity. If omitted, register all identities found in JSON.")
    parser.add_argument("--list-names", action="store_true", help="Print identities found in JSON and exit.")
    parser.add_argument(
        "--identity-mode",
        choices=["first_token", "exact"],
        default="first_token",
        help="first_token maps Simon_front_hat to simon; exact uses the full JSON name.",
    )
    parser.add_argument("--model-path", default="../models/face_recognition.onnx")
    parser.add_argument("--detector-path", default="../models/face_detection_yunet.onnx")
    parser.add_argument("--users-dir", default="../users")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    users_dir = Path(args.users_dir).resolve()

    found = Counter()
    raw_names = defaultdict(Counter)
    for _, raw_name, identity, _ in iter_labelled_faces(dataset_dir, None, args.identity_mode):
        found[identity] += 1
        raw_names[identity][raw_name] += 1

    print("Identities found in JSON:")
    for identity, count in sorted(found.items()):
        examples = ", ".join(name for name, _ in raw_names[identity].most_common(4))
        print(f"- {identity}: {count} faces ({examples})")

    if args.list_names:
        return

    recognizer = RiderRecognizer(
        model_path=args.model_path,
        detector_path=args.detector_path,
        users_dir=users_dir,
    )

    if not recognizer.is_available():
        raise SystemExit(f"Face recognition model not available: {recognizer.model_path}")

    grouped_embeddings = defaultdict(list)
    grouped_used = defaultdict(list)
    for image_path, label_name, identity, bbox in iter_labelled_faces(dataset_dir, args.name, args.identity_mode):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        embedding = recognizer.embedding_from_face(frame, bbox)
        if embedding is None:
            continue
        grouped_embeddings[identity].append(embedding.reshape(-1))
        grouped_used[identity].append((image_path.name, label_name))

    if not grouped_embeddings:
        raise SystemExit("No usable labelled faces found for registration.")

    for identity in sorted(grouped_embeddings):
        embeddings = grouped_embeddings[identity]
        used = grouped_used[identity]
        reference = np.mean(np.stack(embeddings, axis=0), axis=0)
        out_path = users_dir / f"{identity}.npy"
        save_embedding(out_path, reference)

        print(f"Registered rider: {identity}")
        print(f"Reference embedding: {out_path}")
        print(f"Faces used: {len(used)}")
        for filename, label_name in used:
            print(f"- {filename} ({label_name})")


if __name__ == "__main__":
    main()
