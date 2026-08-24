import argparse

import cv2

from project_scan.session import ScanSession
from project_scan.slam.spectacular import (
    SpectacularSlam,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "name",
        help="Zone name",
    )

    parser.add_argument(
        "--low-light",
        action="store_true",
        help="Enable IR flood and low-light VIO exposure",
    )

    args = parser.parse_args()

    backend = SpectacularSlam(
        low_light=args.low_light
    )

    scan = ScanSession(
        args.name,
        backend,
    )

    scan.start()

    print(
        f"Scanning: {args.name}"
    )

    print(
        "Mode: "
        + (
            "low light"
            if args.low_light
            else "normal"
        )
    )

    print(
        "Press Q in the camera window "
        "or Ctrl+C to stop."
    )

    print()

    try:
        while True:
            update = (
                scan.wait_for_update()
            )

            position = (
                update.camera_to_world[
                    :3,
                    3,
                ]
            )

            print(
                "\r"
                f"{update.tracking_status:<14} "
                f"x={position[0]:+.3f} "
                f"y={position[1]:+.3f} "
                f"z={position[2]:+.3f} m  "
                f"keyframes="
                f"{update.mapping.keyframes:<4} "
                f"dense_points="
                f"{update.mapping.dense_points:<8}",
                end="",
                flush=True,
            )

            frame = (
                backend.get_preview_frame()
            )

            if frame is not None:
                cv2.imshow(
                    "Project Scan",
                    frame,
                )

            if (
                cv2.waitKey(1) & 0xFF
                == ord("q")
            ):
                break

    except KeyboardInterrupt:
        pass

    finally:
        cv2.destroyAllWindows()

    print()
    print()
    print(
        "Scan stopped. "
        "Building final map..."
    )
    print()

    zone_dir = scan.stop()

    print()
    print(
        f"Saved zone: "
        f"{zone_dir.resolve()}"
    )


if __name__ == "__main__":
    main()