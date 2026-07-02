from __future__ import annotations

import argparse

import cv2

from app.api.dependencies import get_recognition_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run webcam attendance recognition.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--skip-frames", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = get_recognition_service()
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    frame_index = 0
    latest_names: list[str] = []
    print("Press q to quit.")

    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break

        if frame_index % args.skip_frames == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = service.recognize(frame_rgb, mark_attendance=True)
            latest_names = [match["name"] for match in result["matches"]]

        label = ", ".join(latest_names) if latest_names else "No known face"
        cv2.putText(
            frame_bgr,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Face Attendance", frame_bgr)

        frame_index += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
