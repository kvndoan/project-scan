import math
from importlib.metadata import version

import depthai
import numpy as np
import spectacularAI


def wrap_degrees(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def heading_deg(rotation: np.ndarray) -> float | None:
    # OpenCV camera coordinates use +Z as the forward direction.
    forward = rotation[:, 2]

    horizontal = math.hypot(
        forward[0],
        forward[1],
    )

    # Heading becomes undefined if the camera points almost vertically.
    if horizontal < 0.1:
        return None

    return math.degrees(
        math.atan2(
            forward[1],
            forward[0],
        )
    )


def tilt_deg(rotation: np.ndarray) -> float:
    forward = rotation[:, 2]

    return math.degrees(
        math.atan2(
            forward[2],
            math.hypot(
                forward[0],
                forward[1],
            ),
        )
    )


def rotation_magnitude_deg(
    initial_rotation: np.ndarray,
    rotation: np.ndarray,
) -> float:
    relative = (
        initial_rotation.T
        @ rotation
    )

    cosine = (
        np.trace(relative) - 1.0
    ) / 2.0

    cosine = float(
        np.clip(
            cosine,
            -1.0,
            1.0,
        )
    )

    return math.degrees(
        math.acos(cosine)
    )


def main() -> None:
    print("Project Scan Spectacular AI VIO check")
    print()
    print(
        "Spectacular AI:",
        version("spectacularai"),
    )
    print(
        "DepthAI:",
        version("depthai"),
    )
    print()

    pipeline = depthai.Pipeline()

    vio_pipeline = spectacularAI.depthai.Pipeline(
        pipeline
    )

    print("Starting OAK-D Pro...")
    print()

    with depthai.Device(pipeline) as device, (
        vio_pipeline.startSession(device)
    ) as vio_session:
        print("Spectacular AI VIO started.")
        print()
        print("Suggested test:")
        print("  1. Hold still for ~5 seconds.")
        print("  2. Slide sideways without rotating.")
        print("  3. Hold still.")
        print("  4. Rotate left, return to center.")
        print("  5. Rotate right, return to center.")
        print()
        print("Press Ctrl+C to quit.")
        print()

        initial_position = None
        initial_rotation = None
        initial_heading = None
        initial_tilt = None

        try:
            while True:
                output = vio_session.waitForOutput()

                camera_pose = (
                    output
                    .getCameraPose(0)
                    .pose
                    .asMatrix()
                )

                position = camera_pose[:3, 3]
                rotation = camera_pose[:3, :3]

                if initial_position is None:
                    initial_position = (
                        position.copy()
                    )

                    initial_rotation = (
                        rotation.copy()
                    )

                    initial_heading = heading_deg(
                        rotation
                    )

                    initial_tilt = tilt_deg(
                        rotation
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

                rotation_magnitude = (
                    rotation_magnitude_deg(
                        initial_rotation,
                        rotation,
                    )
                )

                heading = heading_deg(
                    rotation
                )

                if (
                    heading is not None
                    and initial_heading is not None
                ):
                    turn = wrap_degrees(
                        heading
                        - initial_heading
                    )

                    turn_text = (
                        f"{turn:+6.1f}"
                    )
                else:
                    turn_text = "   n/a"

                tilt = (
                    tilt_deg(rotation)
                    - initial_tilt
                )

                angular_velocity = (
                    output.angularVelocity
                )

                angular_speed = math.degrees(
                    math.sqrt(
                        angular_velocity.x ** 2
                        + angular_velocity.y ** 2
                        + angular_velocity.z ** 2
                    )
                )

                status = output.status.name

                print(
                    "\r"
                    f"{status:<14} "
                    f"d=({displacement[0]:+.3f}, "
                    f"{displacement[1]:+.3f}, "
                    f"{displacement[2]:+.3f}) m  "
                    f"dist={distance:.3f} m  "
                    f"turn={turn_text} deg  "
                    f"tilt={tilt:+6.1f} deg  "
                    f"rot_mag={rotation_magnitude:6.1f} deg  "
                    f"ang_vel={angular_speed:6.1f} deg/s",
                    end="",
                    flush=True,
                )

        except KeyboardInterrupt:
            print()
            print()
            print("VIO check stopped.")


if __name__ == "__main__":
    main()