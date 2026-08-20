import numpy as np

from project_scan.slam.spectacular import (
    SpectacularSlam,
)


def main() -> None:
    print("Project Scan SLAM adapter check")
    print()

    with SpectacularSlam() as slam:
        print("SLAM started.")
        print("Move the OAK-D Pro around.")
        print("Press Ctrl+C to stop.")
        print()

        initial_position = None

        try:
            while True:
                update = slam.wait_for_update()

                position = (
                    update.camera_to_world[
                        :3,
                        3,
                    ]
                )

                if initial_position is None:
                    initial_position = (
                        position.copy()
                    )

                displacement = (
                    position
                    - initial_position
                )

                distance = float(
                    np.linalg.norm(
                        displacement
                    )
                )

                print(
                    "\r"
                    f"{update.tracking_status:<14} "
                    f"x={position[0]:+.3f} "
                    f"y={position[1]:+.3f} "
                    f"z={position[2]:+.3f} m  "
                    f"dist={distance:.3f} m  "
                    f"keyframes="
                    f"{update.mapping.keyframes:<4} "
                    f"landmarks="
                    f"{update.mapping.landmarks:<6} "
                    f"dense_points="
                    f"{update.mapping.dense_points:<8}",
                    end="",
                    flush=True,
                )

        except KeyboardInterrupt:
            print()
            print()
            print("SLAM adapter check stopped.")


if __name__ == "__main__":
    main()