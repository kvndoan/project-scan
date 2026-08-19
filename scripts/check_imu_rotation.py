import time

import cv2
import numpy as np

from project_scan.sensors import OakDSensor


BIAS_CALIBRATION_SECONDS = 3.0


def rotation_angle_deg(
    rotation: np.ndarray,
) -> float:
    """
    Return the total rotation angle represented by a rotation matrix.
    """

    cos_angle = (
        np.trace(rotation) - 1.0
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


def rotation_vector_deg(
    rotation: np.ndarray,
) -> np.ndarray:
    """
    Return OpenCV's axis-angle rotation vector in degrees.

    The three values are useful for seeing which IMU axis dominates
    during a controlled rotation.
    """

    rotation_vector, _ = cv2.Rodrigues(
        rotation
    )

    return np.degrees(
        rotation_vector.reshape(3)
    )


def main() -> None:
    with OakDSensor() as sensor:
        print("Project Scan IMU rotation check")
        print()
        print(
            f"Calibrating gyro bias for "
            f"{BIAS_CALIBRATION_SECONDS:.1f} seconds..."
        )
        print("DO NOT MOVE THE CAMERA.")
        print()

        # ------------------------------------------------------------
        # Estimate gyro bias while stationary
        # ------------------------------------------------------------

        bias_samples: list[np.ndarray] = []

        bias_start_time = time.perf_counter()

        while sensor.is_running():
            frame = sensor.poll_frame()

            if frame is None:
                cv2.waitKey(1)
                continue

            for sample in frame.imu_samples:
                bias_samples.append(
                    sample.gyro_rps.copy()
                )

            elapsed_s = (
                time.perf_counter()
                - bias_start_time
            )

            preview = frame.rgb.copy()

            remaining_s = max(
                0.0,
                BIAS_CALIBRATION_SECONDS
                - elapsed_s,
            )

            cv2.putText(
                preview,
                (
                    "GYRO BIAS CALIBRATION - "
                    f"{remaining_s:.1f}s"
                ),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Project Scan IMU Rotation Check",
                preview,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                cv2.destroyAllWindows()
                return

            if elapsed_s >= BIAS_CALIBRATION_SECONDS:
                break

        if not bias_samples:
            raise RuntimeError(
                "No gyro samples were received during calibration"
            )

        gyro_bias_rps = np.mean(
            np.stack(
                bias_samples,
                axis=0,
            ),
            axis=0,
        )

        print(
            "Gyro bias:"
            f" x={gyro_bias_rps[0]:+.6f}"
            f" y={gyro_bias_rps[1]:+.6f}"
            f" z={gyro_bias_rps[2]:+.6f}"
            " rad/s"
        )

        print()
        print("Gyro integration started.")
        print()
        print("Test:")
        print("  0-5 s: leave camera completely still")
        print("  then:  slowly rotate it a small amount")
        print("         in ONE 'shake head NO' direction")
        print("  then:  stop and leave it still for ~10 s")
        print()
        print("The exact angle does NOT matter.")
        print("Press 'q' to quit.")
        print()

        # ------------------------------------------------------------
        # Integrated IMU orientation
        # ------------------------------------------------------------
        #
        # The gyro reports angular velocity in radians per second.
        #
        # For each IMU sample:
        #
        #     angle change = angular velocity * dt
        #
        # We turn that small rotation vector into a rotation matrix
        # and accumulate it.
        # ------------------------------------------------------------

        R_start_current = np.eye(
            3,
            dtype=np.float64,
        )

        previous_gyro_timestamp_s: float | None = None

        test_start_time = time.perf_counter()
        last_status_second = -1

        latest_corrected_gyro_rps = np.zeros(
            3,
            dtype=np.float64,
        )

        integrated_samples = 0
        skipped_samples = 0

        while sensor.is_running():
            frame = sensor.poll_frame()

            if frame is None:
                cv2.waitKey(1)
                continue

            # --------------------------------------------------------
            # Integrate all gyro samples associated with this frame
            # --------------------------------------------------------

            for sample in frame.imu_samples:
                timestamp_s = (
                    sample.gyro_timestamp_s
                )

                if previous_gyro_timestamp_s is None:
                    previous_gyro_timestamp_s = (
                        timestamp_s
                    )
                    continue

                dt_s = (
                    timestamp_s
                    - previous_gyro_timestamp_s
                )

                previous_gyro_timestamp_s = (
                    timestamp_s
                )

                # Gyro is configured for hundreds of Hz.
                #
                # Ignore duplicate timestamps or clearly discontinuous
                # gaps rather than integrating a bad interval.
                if (
                    dt_s <= 0.0
                    or dt_s > 0.05
                ):
                    skipped_samples += 1
                    continue

                corrected_gyro_rps = (
                    sample.gyro_rps
                    - gyro_bias_rps
                )

                latest_corrected_gyro_rps = (
                    corrected_gyro_rps
                )

                # Rotation vector for this tiny timestep.
                delta_rotation_vector = (
                    corrected_gyro_rps
                    * dt_s
                )

                delta_rotation, _ = (
                    cv2.Rodrigues(
                        delta_rotation_vector
                    )
                )

                # Gyro angular velocity is measured in the current
                # sensor/body frame, so compose the incremental
                # rotation on the right.
                R_start_current = (
                    R_start_current
                    @ delta_rotation
                )

                integrated_samples += 1

            # --------------------------------------------------------
            # Diagnostics
            # --------------------------------------------------------

            elapsed_s = (
                time.perf_counter()
                - test_start_time
            )

            total_rotation_deg = (
                rotation_angle_deg(
                    R_start_current
                )
            )

            rotation_vector = (
                rotation_vector_deg(
                    R_start_current
                )
            )

            gyro_dps = np.degrees(
                latest_corrected_gyro_rps
            )

            preview = frame.rgb.copy()

            cv2.putText(
                preview,
                (
                    "IMU rotation: "
                    f"{total_rotation_deg:.1f} deg"
                ),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                preview,
                (
                    "Rot vec: "
                    f"x={rotation_vector[0]:+.1f}  "
                    f"y={rotation_vector[1]:+.1f}  "
                    f"z={rotation_vector[2]:+.1f} deg"
                ),
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                preview,
                (
                    "Gyro: "
                    f"x={gyro_dps[0]:+.1f}  "
                    f"y={gyro_dps[1]:+.1f}  "
                    f"z={gyro_dps[2]:+.1f} deg/s"
                ),
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                preview,
                f"Test time: {elapsed_s:.1f}s",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Project Scan IMU Rotation Check",
                preview,
            )

            # --------------------------------------------------------
            # Terminal output once per second
            # --------------------------------------------------------

            current_second = int(
                elapsed_s
            )

            if current_second != last_status_second:
                last_status_second = current_second

                print(
                    f"t={elapsed_s:5.1f}s | "
                    f"imu_rot={total_rotation_deg:6.2f} deg | "
                    f"rotvec=("
                    f"{rotation_vector[0]:+7.2f}, "
                    f"{rotation_vector[1]:+7.2f}, "
                    f"{rotation_vector[2]:+7.2f}) deg | "
                    f"gyro=("
                    f"{gyro_dps[0]:+7.2f}, "
                    f"{gyro_dps[1]:+7.2f}, "
                    f"{gyro_dps[2]:+7.2f}) deg/s | "
                    f"samples={integrated_samples} | "
                    f"skipped={skipped_samples}"
                )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()