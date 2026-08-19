import time

import cv2
import numpy as np

from project_scan.sensors import OakDSensor


# Keep this consistent with the current Open3D odometry depth cutoff.
USABLE_DEPTH_MAX_MM = 4000

# Visualization only.
DISPLAY_DEPTH_MIN_MM = 250
DISPLAY_DEPTH_MAX_MM = USABLE_DEPTH_MAX_MM

PRINT_INTERVAL_S = 1.0


def percentage(mask: np.ndarray) -> float:
    """
    Return the percentage of True values in a boolean mask.
    """

    if mask.size == 0:
        return 0.0

    return float(
        100.0
        * np.count_nonzero(mask)
        / mask.size
    )


def center_region(
    image: np.ndarray,
) -> np.ndarray:
    """
    Return the middle 50% of the image in both dimensions.
    """

    height, width = image.shape[:2]

    y0 = height // 4
    y1 = 3 * height // 4

    x0 = width // 4
    x1 = 3 * width // 4

    return image[
        y0:y1,
        x0:x1,
    ]


def colorize_depth(
    depth_mm: np.ndarray,
) -> np.ndarray:
    """
    Convert the uint16 depth image into a color visualization.

    Invalid depth values remain black.
    """

    valid = depth_mm > 0

    clipped = np.clip(
        depth_mm,
        DISPLAY_DEPTH_MIN_MM,
        DISPLAY_DEPTH_MAX_MM,
    )

    normalized = (
        (
            clipped.astype(np.float32)
            - DISPLAY_DEPTH_MIN_MM
        )
        / (
            DISPLAY_DEPTH_MAX_MM
            - DISPLAY_DEPTH_MIN_MM
        )
        * 255.0
    )

    normalized = np.clip(
        normalized,
        0,
        255,
    ).astype(np.uint8)

    # Reverse it so closer geometry is visually distinct.
    normalized = 255 - normalized

    depth_color = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_TURBO,
    )

    depth_color[~valid] = 0

    return depth_color


def main() -> None:
    with OakDSensor() as sensor:
        print("Project Scan depth quality check")
        print()
        print("Black pixels represent invalid depth.")
        print(
            f"'Usable' means 0 < depth <= "
            f"{USABLE_DEPTH_MAX_MM / 1000:.1f} m."
        )
        print()
        print("Press 'q' to quit.")

        last_print_time = 0.0

        while sensor.is_running():
            frame = sensor.poll_frame()

            if frame is None:
                cv2.waitKey(1)
                continue

            depth_mm = frame.depth_mm

            # --------------------------------------------------------
            # Full-image validity
            # --------------------------------------------------------

            valid_mask = (
                depth_mm > 0
            )

            usable_mask = (
                valid_mask
                & (
                    depth_mm
                    <= USABLE_DEPTH_MAX_MM
                )
            )

            valid_pct = percentage(
                valid_mask
            )

            usable_pct = percentage(
                usable_mask
            )

            # --------------------------------------------------------
            # Center-region validity
            # --------------------------------------------------------

            center_depth = center_region(
                depth_mm
            )

            center_valid_mask = (
                center_depth > 0
            )

            center_usable_mask = (
                center_valid_mask
                & (
                    center_depth
                    <= USABLE_DEPTH_MAX_MM
                )
            )

            center_valid_pct = percentage(
                center_valid_mask
            )

            center_usable_pct = percentage(
                center_usable_mask
            )

            # --------------------------------------------------------
            # Depth distribution
            # --------------------------------------------------------

            usable_depth_values = (
                depth_mm[
                    usable_mask
                ]
            )

            if usable_depth_values.size > 0:
                p10_mm = float(
                    np.percentile(
                        usable_depth_values,
                        10,
                    )
                )

                median_mm = float(
                    np.median(
                        usable_depth_values
                    )
                )

                p90_mm = float(
                    np.percentile(
                        usable_depth_values,
                        90,
                    )
                )
            else:
                p10_mm = 0.0
                median_mm = 0.0
                p90_mm = 0.0

            # --------------------------------------------------------
            # Visualization
            # --------------------------------------------------------

            rgb_preview = frame.rgb.copy()

            depth_preview = colorize_depth(
                depth_mm
            )

            cv2.putText(
                depth_preview,
                f"Valid depth: {valid_pct:.1f}%",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                depth_preview,
                (
                    f"Usable <= 4m: "
                    f"{usable_pct:.1f}%"
                ),
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                depth_preview,
                (
                    "Center valid: "
                    f"{center_valid_pct:.1f}%"
                ),
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                depth_preview,
                (
                    "Center <= 4m: "
                    f"{center_usable_pct:.1f}%"
                ),
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                depth_preview,
                (
                    "Depth p10/median/p90: "
                    f"{p10_mm / 1000:.2f} / "
                    f"{median_mm / 1000:.2f} / "
                    f"{p90_mm / 1000:.2f} m"
                ),
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Draw the center region used for the center statistics.
            height, width = depth_preview.shape[:2]

            cv2.rectangle(
                depth_preview,
                (
                    width // 4,
                    height // 4,
                ),
                (
                    3 * width // 4,
                    3 * height // 4,
                ),
                (255, 255, 255),
                2,
            )

            # Resize only for display.
            rgb_display = cv2.resize(
                rgb_preview,
                (640, 480),
                interpolation=cv2.INTER_AREA,
            )

            depth_display = cv2.resize(
                depth_preview,
                (640, 480),
                interpolation=cv2.INTER_NEAREST,
            )

            combined = np.hstack(
                (
                    rgb_display,
                    depth_display,
                )
            )

            cv2.imshow(
                "Project Scan Depth Quality",
                combined,
            )

            # --------------------------------------------------------
            # Terminal output once per second
            # --------------------------------------------------------

            now = time.perf_counter()

            if (
                now - last_print_time
                >= PRINT_INTERVAL_S
            ):
                last_print_time = now

                print(
                    f"valid={valid_pct:5.1f}% | "
                    f"usable<=4m={usable_pct:5.1f}% | "
                    f"center_valid={center_valid_pct:5.1f}% | "
                    f"center<=4m={center_usable_pct:5.1f}% | "
                    f"median={median_mm / 1000:.2f}m"
                )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()