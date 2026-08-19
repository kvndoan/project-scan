import time

import cv2
import numpy as np

from project_scan.pose import Open3DRGBDOdometry
from project_scan.sensors import OakDSensor


WARMUP_SECONDS = 3.0


def rotation_angle_deg(
    rotation: np.ndarray,
) -> float:
    """
    Return how far a rotation matrix is rotated from identity.
    """

    cos_angle = (
        np.trace(rotation)
        - 1.0
    ) / 2.0

    cos_angle = np.clip(
        cos_angle,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(
                cos_angle
            )
        )
    )


def main() -> None:
    estimator = Open3DRGBDOdometry()

    with OakDSensor() as sensor:
        print(
            "Project Scan gyro-assisted "
            "RGB-D odometry check"
        )

        print()

        print(
            f"Warming up camera for "
            f"{WARMUP_SECONDS:.1f} seconds..."
        )

        print(
            "Do not move the camera."
        )

        print()

        # ------------------------------------------------------------
        # Camera warm-up
        # ------------------------------------------------------------

        warmup_start = (
            time.perf_counter()
        )

        while sensor.is_running():
            frame = (
                sensor.poll_frame()
            )

            if frame is None:
                cv2.waitKey(1)
                continue

            elapsed_warmup_s = (
                time.perf_counter()
                - warmup_start
            )

            preview = (
                frame.rgb.copy()
            )

            remaining_s = max(
                0.0,
                WARMUP_SECONDS
                - elapsed_warmup_s,
            )

            cv2.putText(
                preview,
                (
                    "WARMING UP - "
                    f"{remaining_s:.1f}s"
                ),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Project Scan Odometry Check",
                preview,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                cv2.destroyAllWindows()
                return

            if (
                elapsed_warmup_s
                >= WARMUP_SECONDS
            ):
                break

        # ------------------------------------------------------------
        # Establish world origin after warm-up
        # ------------------------------------------------------------

        estimator.reset(
            sensor.intrinsics
        )

        print(
            "Warm-up complete."
        )

        print(
            "Odometry started."
        )

        print()

        print(
            "Rotation validation:"
        )

        print(
            "  1. Leave camera completely "
            "still for ~5 s."
        )

        print(
            "  2. Slowly rotate it a SMALL "
            "amount in one 'NO' direction."
        )

        print(
            "  3. Do not rotate back."
        )

        print(
            "  4. Leave it completely still "
            "for ~10 s."
        )

        print()

        print(
            "Exact physical angle is not required."
        )

        print(
            "Press 'q' to quit."
        )

        test_start_time: (
            float | None
        ) = None

        last_status_second = -1

        while sensor.is_running():
            frame = (
                sensor.poll_frame()
            )

            if frame is None:
                cv2.waitKey(1)
                continue

            if test_start_time is None:
                test_start_time = (
                    time.perf_counter()
                )

            elapsed_s = (
                time.perf_counter()
                - test_start_time
            )

            start_time = (
                time.perf_counter()
            )

            pose = estimator.update(
                frame
            )

            odometry_ms = (
                time.perf_counter()
                - start_time
            ) * 1000.0

            diagnostics = (
                estimator.last_diagnostics
            )

            position = (
                pose.T_world_camera[
                    :3,
                    3,
                ]
            )

            rotation = (
                pose.T_world_camera[
                    :3,
                    :3,
                ]
            )

            global_rotation_deg = (
                rotation_angle_deg(
                    rotation
                )
            )

            (
                relative_x,
                relative_y,
                relative_z,
            ) = (
                diagnostics
                .relative_translation_xyz_m
            )

            preview = (
                frame.rgb.copy()
            )

            tracking_text = (
                "TRACKING"
                if pose.tracking_ok
                else "REJECTED"
            )

            tracking_color = (
                (0, 255, 0)
                if pose.tracking_ok
                else (0, 0, 255)
            )

            # --------------------------------------------------------
            # Tracking state
            # --------------------------------------------------------

            cv2.putText(
                preview,
                (
                    f"Tracking: "
                    f"{tracking_text}"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                tracking_color,
                2,
                cv2.LINE_AA,
            )

            # --------------------------------------------------------
            # Global position
            # --------------------------------------------------------

            cv2.putText(
                preview,
                (
                    "Global: "
                    f"x={position[0]:+.3f}  "
                    f"y={position[1]:+.3f}  "
                    f"z={position[2]:+.3f} m"
                ),
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # --------------------------------------------------------
            # Global rotation
            # --------------------------------------------------------

            cv2.putText(
                preview,
                (
                    "Global rotation: "
                    f"{global_rotation_deg:.1f} deg"
                ),
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # --------------------------------------------------------
            # Open3D relative estimate
            # --------------------------------------------------------

            cv2.putText(
                preview,
                (
                    "Open3D step: "
                    f"{diagnostics.relative_translation_m:.3f} m  "
                    f"{diagnostics.relative_rotation_deg:.2f} deg"
                ),
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # --------------------------------------------------------
            # Gyro prior
            # --------------------------------------------------------

            cv2.putText(
                preview,
                (
                    "IMU prior: "
                    f"{diagnostics.imu_prior_rotation_deg:.2f} deg  "
                    f"samples={diagnostics.imu_samples_used}"
                ),
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # --------------------------------------------------------
            # Rotation disagreement
            # --------------------------------------------------------

            cv2.putText(
                preview,
                (
                    "IMU/Open3D rot error: "
                    f"{diagnostics.imu_open3d_rotation_error_deg:.2f} deg"
                ),
                (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # --------------------------------------------------------
            # Signed translation
            # --------------------------------------------------------

            cv2.putText(
                preview,
                (
                    "Step xyz: "
                    f"{relative_x:+.3f}  "
                    f"{relative_y:+.3f}  "
                    f"{relative_z:+.3f} m"
                ),
                (20, 245),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # --------------------------------------------------------
            # Timing
            # --------------------------------------------------------

            cv2.putText(
                preview,
                (
                    f"dt="
                    f"{diagnostics.dt_s * 1000:.0f} ms  "
                    f"gap={diagnostics.frame_gap}  "
                    f"odo={odometry_ms:.1f} ms  "
                    f"test={elapsed_s:.1f}s"
                ),
                (20, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if not pose.tracking_ok:
                reason = (
                    diagnostics.rejection_reason
                    or "unknown"
                )

                cv2.putText(
                    preview,
                    f"Reason: {reason}",
                    (20, 315),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(
                "Project Scan Odometry Check",
                preview,
            )

            # --------------------------------------------------------
            # One terminal status line per second
            # --------------------------------------------------------

            current_second = int(
                elapsed_s
            )

            if (
                current_second
                != last_status_second
            ):
                last_status_second = (
                    current_second
                )

                print(
                    f"t={elapsed_s:5.1f}s | "
                    f"global=("
                    f"{position[0]:+.3f}, "
                    f"{position[1]:+.3f}, "
                    f"{position[2]:+.3f}) m | "
                    f"rot={global_rotation_deg:5.1f} deg | "
                    f"step_rot="
                    f"{diagnostics.relative_rotation_deg:5.2f} deg | "
                    f"imu_prior="
                    f"{diagnostics.imu_prior_rotation_deg:5.2f} deg | "
                    f"imu_err="
                    f"{diagnostics.imu_open3d_rotation_error_deg:5.2f} deg | "
                    f"imu_n="
                    f"{diagnostics.imu_samples_used:2d} | "
                    f"step=("
                    f"{relative_x:+.4f}, "
                    f"{relative_y:+.4f}, "
                    f"{relative_z:+.4f}) m | "
                    f"{tracking_text}"
                )

            # --------------------------------------------------------
            # Always log rejected estimates
            # --------------------------------------------------------

            if not pose.tracking_ok:
                print(
                    "REJECTED | "
                    f"gap={diagnostics.frame_gap} | "
                    f"dt={diagnostics.dt_s:.3f}s | "
                    f"translation="
                    f"{diagnostics.relative_translation_m:.3f}m | "
                    f"open3d_rot="
                    f"{diagnostics.relative_rotation_deg:.1f}deg | "
                    f"imu_prior="
                    f"{diagnostics.imu_prior_rotation_deg:.2f}deg | "
                    f"imu_error="
                    f"{diagnostics.imu_open3d_rotation_error_deg:.1f}deg | "
                    f"imu_n="
                    f"{diagnostics.imu_samples_used} | "
                    f"reason="
                    f"{diagnostics.rejection_reason}"
                )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()