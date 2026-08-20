import argparse

from project_scan.session import ScanSession
from project_scan.slam.spectacular import SpectacularSlam


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Zone name")
    args = parser.parse_args()

    scan = ScanSession(args.name, SpectacularSlam())
    scan.start()

    print(f"Scanning: {args.name}")
    print("Press Ctrl+C to stop and save.")
    print()

    try:
        while True:
            update = scan.wait_for_update()
            position = update.camera_to_world[:3, 3]

            print(
                "\r"
                f"{update.tracking_status:<14} "
                f"x={position[0]:+.3f} "
                f"y={position[1]:+.3f} "
                f"z={position[2]:+.3f} m  "
                f"keyframes={update.mapping.keyframes:<4} "
                f"dense_points={update.mapping.dense_points:<8}",
                end="",
                flush=True,
            )
    except KeyboardInterrupt:
        print()
        print()

    zone_dir = scan.stop()
    print(f"Saved zone: {zone_dir.resolve()}")


if __name__ == "__main__":
    main()