import cv2
import numpy as np

from project_scan.sensors import OakDSensor


def colorize_depth(
    depth_mm: np.ndarray,
) -> np.ndarray:
    """
    Convert a millimeter depth image into a viewable color image.
    """

    invalid_mask = depth_mm == 0

    valid_depths = depth_mm[~invalid_mask]

    if valid_depths.size == 0:
        return np.zeros(
            (*depth_mm.shape, 3),
            dtype=np.uint8,
        )

    min_depth = np.percentile(
        valid_depths,
        3,
    )

    max_depth = np.percentile(
        valid_depths,
        95,
    )

    log_depth = np.zeros_like(
        depth_mm,
        dtype=np.float32,
    )

    np.log(
        depth_mm,
        where=depth_mm != 0,
        out=log_depth,
    )

    log_min = np.log(min_depth)
    log_max = np.log(max_depth)

    log_depth = np.clip(
        log_depth,
        log_min,
        log_max,
    )

    normalized = np.interp(
        log_depth,
        (log_min, log_max),
        (0, 255),
    ).astype(np.uint8)

    colorized = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_TURBO,
    )

    colorized[invalid_mask] = 0

    return colorized


def main() -> None:
    with OakDSensor() as sensor:
        intrinsics = sensor.intrinsics

        print("Project Scan sensor check")

        print(
            "RGB intrinsics: "
            f"fx={intrinsics.fx:.2f}, "
            f"fy={intrinsics.fy:.2f}, "
            f"cx={intrinsics.cx:.2f}, "
            f"cy={intrinsics.cy:.2f}"
        )

        print("Press 'q' to quit.")

        while sensor.is_running():
            frame = sensor.poll_frame()

            if frame is None:
                cv2.waitKey(1)
                continue

            depth_view = colorize_depth(
                frame.depth_mm
            )

            preview = cv2.addWeighted(
                frame.rgb,
                0.6,
                depth_view,
                0.4,
                0,
            )

            status = (
                f"frame={frame.sequence_num}  "
                f"t={frame.timestamp_s:.3f}s  "
                f"imu={len(frame.imu_samples)}"
            )

            cv2.putText(
                preview,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Project Scan Sensor Check",
                preview,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()