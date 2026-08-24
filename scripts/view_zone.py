import argparse
import time
from pathlib import Path

import numpy as np
import open3d as o3d


KEY_UP = 265
KEY_DOWN = 264
KEY_LEFT = 263
KEY_RIGHT = 262
KEY_LEFT_CTRL = 341
KEY_RIGHT_CTRL = 345

RELEASE = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zone", type=Path)
    args = parser.parse_args()

    map_path = args.zone / "map.ply"
    trajectory_path = args.zone / "trajectory.npz"

    if not map_path.exists():
        raise FileNotFoundError(map_path)

    if not trajectory_path.exists():
        raise FileNotFoundError(trajectory_path)

    point_cloud = o3d.io.read_point_cloud(str(map_path))

    point_cloud, _ = point_cloud.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0,
    )

    if not point_cloud.has_points():
        raise RuntimeError("Map contains no points")

    print(f"Loaded {len(point_cloud.points):,} map points")

    with np.load(trajectory_path) as data:
        positions = data["camera_to_world"][:, :3, 3]

    viewer = o3d.visualization.VisualizerWithKeyCallback()
    viewer.create_window(
        window_name=f"Project Scan - {args.zone.name}"
    )

    # Let the map establish the viewer bounding box.
    viewer.add_geometry(point_cloud)

    if len(positions) >= 2:
        lines = np.column_stack(
            (
                np.arange(len(positions) - 1),
                np.arange(1, len(positions)),
            )
        )

        trajectory = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(positions),
            lines=o3d.utility.Vector2iVector(lines),
        )

        trajectory.colors = o3d.utility.Vector3dVector(
            np.tile(
                [1.0, 0.0, 0.0],
                (len(lines), 1),
            )
        )

        # Don't let trajectory outliers change the map framing.
        viewer.add_geometry(
            trajectory,
            reset_bounding_box=False,
        )

    render_option = viewer.get_render_option()
    render_option.point_size = 2.0

    center = (
        point_cloud
        .get_axis_aligned_bounding_box()
        .get_center()
    )

    def level_view(viewer) -> bool:
        view = viewer.get_view_control()

        # Spectacular AI uses Z-up.
        # Front must not be parallel to up.
        view.set_lookat(center)
        view.set_front([0.0, -1.0, -0.25])
        view.set_up([0.0, 0.0, 1.0])
        view.set_zoom(0.7)
        view.set_constant_z_near(0.02)

        return False

    # Establish a valid initial camera immediately.
    level_view(viewer)

    held_keys: set[int] = set()

    def key_action(key: int):
        def callback(viewer, action, modifiers):
            if action == RELEASE:
                held_keys.discard(key)
            else:
                held_keys.add(key)

            return False

        return callback

    movement_keys = [
        ord("W"),
        ord("A"),
        ord("S"),
        ord("D"),
        ord(" "),
        ord("I"),
        ord("J"),
        ord("K"),
        ord("L"),
        KEY_UP,
        KEY_DOWN,
        KEY_LEFT,
        KEY_RIGHT,
        KEY_LEFT_CTRL,
        KEY_RIGHT_CTRL,
    ]

    for key in movement_keys:
        viewer.register_key_action_callback(
            key,
            key_action(key),
        )

    viewer.register_key_callback(
        ord("R"),
        level_view,
    )

    previous_time = time.perf_counter()

    move_speed = 1.0
    look_speed = 350.0

    def update_camera(viewer) -> bool:
        nonlocal previous_time

        now = time.perf_counter()
        dt = min(now - previous_time, 0.1)
        previous_time = now

        forward = (
            int(
                ord("W") in held_keys
                or KEY_UP in held_keys
            )
            - int(
                ord("S") in held_keys
                or KEY_DOWN in held_keys
            )
        )

        right = (
            int(
                ord("D") in held_keys
                or KEY_RIGHT in held_keys
            )
            - int(
                ord("A") in held_keys
                or KEY_LEFT in held_keys
            )
        )

        up = (
            int(ord(" ") in held_keys)
            - int(
                KEY_LEFT_CTRL in held_keys
                or KEY_RIGHT_CTRL in held_keys
            )
        )

        yaw = (
            int(ord("L") in held_keys)
            - int(ord("J") in held_keys)
        )

        pitch = (
            int(ord("K") in held_keys)
            - int(ord("I") in held_keys)
        )

        view = viewer.get_view_control()

        if forward or right or up:
            distance = move_speed * dt

            view.camera_local_translate(
                forward * distance,
                right * distance,
                up * distance,
            )

        if yaw or pitch:
            rotation = look_speed * dt

            view.camera_local_rotate(
                yaw * rotation,
                pitch * rotation,
            )

        return False

    viewer.register_animation_callback(
        update_camera
    )

    print()
    print("Controls:")
    print("  W / Up       forward")
    print("  S / Down     backward")
    print("  A / Left     left")
    print("  D / Right    right")
    print("  Space        up")
    print("  Ctrl         down")
    print("  I / K        look up / down")
    print("  J / L        look left / right")
    print("  R            re-center + level")
    print("  Mouse        normal Open3D controls")

    viewer.run()
    viewer.destroy_window()


if __name__ == "__main__":
    main()