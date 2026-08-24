import argparse
import time

import cv2
import numpy as np
import open3d as o3d

from project_scan.session import ScanSession
from project_scan.slam.spectacular import (
    SpectacularSlam,
)


STATUS_INTERVAL_S = 0.2
LIVE_MAP_INTERVAL_S = 0.25
LIVE_RENDER_INTERVAL_S = 0.1


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "name",
        help="Zone name",
    )

    parser.add_argument(
        "--low-light",
        action="store_true",
        help=(
            "Enable IR flood and "
            "low-light VIO exposure"
        ),
    )

    parser.add_argument(
        "--live-map",
        action="store_true",
        help=(
            "Show the growing 3D "
            "point cloud while scanning"
        ),
    )

    args = parser.parse_args()

    backend = SpectacularSlam(
        low_light=args.low_light,
        live_map=args.live_map,
    )

    scan = ScanSession(
        args.name,
        backend,
    )

    live_viewer = None
    live_cloud = None
    live_cloud_added = False

    if args.live_map:
        live_viewer = (
            o3d.visualization
            .VisualizerWithKeyCallback()
        )

        live_viewer.create_window(
            window_name=(
                "Project Scan - Live Map"
            ),
            width=800,
            height=600,
        )

        live_cloud = (
            o3d.geometry.PointCloud()
        )

        live_viewer.get_render_option(
        ).point_size = 2.0

        def recenter(viewer) -> bool:
            if (
                not live_cloud_added
                or live_cloud is None
            ):
                return False

            center = (
                live_cloud
                .get_axis_aligned_bounding_box()
                .get_center()
            )

            view = (
                viewer.get_view_control()
            )

            view.set_lookat(
                center
            )

            view.set_front(
                [0.0, -1.0, -0.3]
            )

            view.set_up(
                [0.0, 0.0, 1.0]
            )

            view.set_zoom(
                0.7
            )

            return False

        live_viewer.register_key_callback(
            ord("R"),
            recenter,
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

    if args.live_map:
        print(
            "Live map: enabled "
            "(R = recenter)"
        )

    print()

    next_status = 0.0
    next_live_map = 0.0
    next_live_render = 0.0

    try:
        while True:
            update = (
                scan.wait_for_update()
            )

            now = time.monotonic()

            if now >= next_status:
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

                next_status = (
                    now
                    + STATUS_INTERVAL_S
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
                live_viewer is not None
                and live_cloud is not None
            ):
                if now >= next_live_map:
                    live_map = (
                        backend
                        .get_live_point_cloud()
                    )

                    if live_map is not None:
                        points, colors = (
                            live_map
                        )

                        live_cloud.points = (
                            o3d.utility
                            .Vector3dVector(
                                points
                            )
                        )

                        if colors is not None:
                            live_cloud.colors = (
                                o3d.utility
                                .Vector3dVector(
                                    colors.astype(
                                        np.float64
                                    )
                                    / 255.0
                                )
                            )

                        if not live_cloud_added:
                            live_viewer.add_geometry(
                                live_cloud
                            )

                            live_cloud_added = True

                            recenter(
                                live_viewer
                            )

                        else:
                            live_viewer.update_geometry(
                                live_cloud
                            )

                    next_live_map = (
                        now
                        + LIVE_MAP_INTERVAL_S
                    )

                if now >= next_live_render:
                    if not live_viewer.poll_events():
                        live_viewer.destroy_window()
                        live_viewer = None
                        live_cloud = None

                    else:
                        live_viewer.update_renderer()

                    next_live_render = (
                        now
                        + LIVE_RENDER_INTERVAL_S
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

        if live_viewer is not None:
            live_viewer.destroy_window()

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