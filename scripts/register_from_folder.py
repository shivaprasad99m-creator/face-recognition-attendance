from __future__ import annotations

import argparse
from pathlib import Path

from app.api.dependencies import get_recognition_service
from app.utils.image_io import decode_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register users from folders named '<external_id>__<name>'."
    )
    parser.add_argument("dataset", type=Path, help="Folder containing one subfolder per user")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = get_recognition_service()
    image_extensions = {".jpg", ".jpeg", ".png"}

    for user_dir in sorted(path for path in args.dataset.iterdir() if path.is_dir()):
        if "__" not in user_dir.name:
            print(f"Skipping {user_dir.name}: expected '<external_id>__<name>'")
            continue
        external_id, name = user_dir.name.split("__", maxsplit=1)
        for image_path in sorted(user_dir.iterdir()):
            if image_path.suffix.lower() not in image_extensions:
                continue
            try:
                rgb_image = decode_image(image_path.read_bytes())
                result = service.register_face(external_id, name.replace("_", " "), rgb_image)
                print(
                    f"Registered {result['user']['name']} from {image_path.name} "
                    f"(embedding_id={result['embedding_id']})"
                )
            except Exception as exc:
                print(f"Failed {image_path}: {exc}")


if __name__ == "__main__":
    main()
