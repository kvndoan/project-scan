from contextlib import ExitStack
from pathlib import Path
from threading import Lock
import shutil
import subprocess

import depthai
import numpy as np
import spectacularAI

from .base import MappingStats, SlamBackend, SlamUpdate


IR_DOT_INTENSITY = 0.5
IR_FLOOD_INTENSITY = 0.3
KEYFRAME_DISTANCE_M = 0.15
LIVE_MAP_MAX_POINTS = 100_000


class SpectacularSlam(SlamBackend):
    def __init__(
        self,
        low_light: bool = False,
        live_map: bool = False,
    ) -> None:
        self._low_light = low_light
        self._live_map_enabled = live_map

        self._resources: ExitStack | None = None
        self._session = None
        self._preview_queue = None
        self._recording_path: Path | None = None
        self._mapping = MappingStats()

        self._live_keyframes: dict[
            int,
            tuple[np.ndarray, np.ndarray | None],
        ] = {}

        self._live_map_lock = Lock()
        self._live_map_version = 0
        self._shown_live_map_version = -1

    def start(
        self,
        recording_path: Path | None = None,
    ) -> None:
        if self._resources is not None:
            raise RuntimeError(
                "SLAM backend is already started"
            )

        pipeline = depthai.Pipeline()

        config = spectacularAI.depthai.Configuration()

        config.internalParameters = {
            "extendParameterSets": [
                "point-cloud"
            ]
        }

        config.useVioAutoExposure = self._low_light

        if recording_path is not None:
            config.recordingFolder = str(
                recording_path
            )

        vio_pipeline = spectacularAI.depthai.Pipeline(
            pipeline,
            config,
            self._on_mapping_output,
        )

        preview_out = pipeline.create(
            depthai.node.XLinkOut
        )

        preview_out.setStreamName(
            "preview"
        )

        preview_out.input.setBlocking(
            False
        )

        preview_out.input.setQueueSize(
            1
        )

        vio_pipeline.monoPrimary.out.link(
            preview_out.input
        )

        resources = ExitStack()

        try:
            device = resources.enter_context(
                depthai.Device(pipeline)
            )

            session = resources.enter_context(
                vio_pipeline.startSession(
                    device
                )
            )

            device.setIrLaserDotProjectorIntensity(
                IR_DOT_INTENSITY
            )

            if self._low_light:
                device.setIrFloodLightIntensity(
                    IR_FLOOD_INTENSITY
                )

            preview_queue = device.getOutputQueue(
                name="preview",
                maxSize=1,
                blocking=False,
            )

        except Exception:
            resources.close()
            raise

        self._mapping = MappingStats()
        self._live_keyframes = {}
        self._live_map_version = 0
        self._shown_live_map_version = -1

        self._recording_path = recording_path
        self._resources = resources
        self._session = session
        self._preview_queue = preview_queue

    def stop(self) -> None:
        if self._resources is None:
            return

        self._resources.close()

        self._resources = None
        self._session = None
        self._preview_queue = None

    def wait_for_update(
        self,
    ) -> SlamUpdate:
        if self._session is None:
            raise RuntimeError(
                "SLAM backend has not been started"
            )

        output = self._session.waitForOutput()

        camera_pose = output.getCameraPose(
            0
        ).pose

        camera_to_world = np.array(
            camera_pose.asMatrix(),
            dtype=np.float64,
            copy=True,
        )

        return SlamUpdate(
            timestamp=camera_pose.time,
            tracking_status=output.status.name,
            camera_to_world=camera_to_world,
            mapping=self._mapping,
        )

    def get_preview_frame(
        self,
    ) -> np.ndarray | None:
        if self._preview_queue is None:
            return None

        frame = self._preview_queue.tryGet()

        if frame is None:
            return None

        return np.array(
            frame.getCvFrame(),
            copy=True,
        )

    def get_live_point_cloud(
        self,
    ) -> (
        tuple[np.ndarray, np.ndarray | None]
        | None
    ):
        if not self._live_map_enabled:
            return None

        with self._live_map_lock:
            if (
                self._live_map_version
                == self._shown_live_map_version
            ):
                return None

            keyframes = list(
                self._live_keyframes.values()
            )

            version = self._live_map_version

        if not keyframes:
            return None

        points = np.concatenate(
            [
                item[0]
                for item in keyframes
            ]
        )

        have_colors = all(
            item[1] is not None
            for item in keyframes
        )

        colors = None

        if have_colors:
            colors = np.concatenate(
                [
                    item[1]
                    for item in keyframes
                ]
            )

        if len(points) > LIVE_MAP_MAX_POINTS:
            step = int(
                np.ceil(
                    len(points)
                    / LIVE_MAP_MAX_POINTS
                )
            )

            points = points[::step]

            if colors is not None:
                colors = colors[::step]

        self._shown_live_map_version = version

        return points, colors

    def export_map(
        self,
        path: Path,
    ) -> None:
        if self._resources is not None:
            raise RuntimeError(
                "Stop SLAM before exporting the map"
            )

        if self._recording_path is None:
            raise RuntimeError(
                "No recording is available"
            )

        sai_cli = shutil.which(
            "sai-cli"
        )

        if sai_cli is None:
            raise RuntimeError(
                "sai-cli was not found"
            )

        subprocess.run(
            [
                sai_cli,
                "process",
                str(self._recording_path),
                "--device_preset=oak-d",
                (
                    "--key_frame_distance="
                    f"{KEYFRAME_DISTANCE_M}"
                ),
                str(path),
            ],
            check=True,
        )

    def _on_mapping_output(
        self,
        output,
    ) -> None:
        slam_map = output.map

        if self._live_map_enabled:
            updated = {}

            for frame_id in (
                output.updatedKeyFrames
            ):
                keyframe = (
                    slam_map.keyFrames.get(
                        frame_id
                    )
                )

                if (
                    keyframe is None
                    or keyframe.pointCloud is None
                ):
                    continue

                frame_set = keyframe.frameSet

                target_frame = (
                    frame_set.rgbFrame
                    or frame_set.primaryFrame
                )

                if target_frame is None:
                    continue

                positions = np.array(
                    keyframe.pointCloud
                    .getPositionData(),
                    dtype=np.float64,
                    copy=True,
                )

                if positions.size == 0:
                    continue

                camera_to_world = np.asarray(
                    target_frame
                    .cameraPose
                    .getCameraToWorldMatrix(),
                    dtype=np.float64,
                )

                positions = (
                    positions
                    @ camera_to_world[
                        :3,
                        :3,
                    ].T
                    + camera_to_world[
                        :3,
                        3,
                    ]
                )

                colors = None

                if (
                    keyframe.pointCloud
                    .hasColors()
                ):
                    colors = np.array(
                        keyframe.pointCloud
                        .getRGB24Data(),
                        dtype=np.uint8,
                        copy=True,
                    )

                updated[frame_id] = (
                    positions,
                    colors,
                )

            if updated:
                with self._live_map_lock:
                    self._live_keyframes.update(
                        updated
                    )

                    self._live_map_version += 1

        dense_points = sum(
            keyframe.pointCloud.size()
            for keyframe
            in slam_map.keyFrames.values()
            if keyframe.pointCloud is not None
        )

        self._mapping = MappingStats(
            keyframes=len(
                slam_map.keyFrames
            ),
            landmarks=len(
                slam_map.mapPoints
            ),
            dense_points=dense_points,
        )