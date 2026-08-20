from contextlib import ExitStack
from pathlib import Path

import depthai
import numpy as np
import spectacularAI

from .base import MappingStats, SlamBackend, SlamUpdate


class SpectacularSlam(SlamBackend):
    def __init__(self) -> None:
        self._resources: ExitStack | None = None
        self._session = None
        self._preview_queue = None
        self._mapping = MappingStats()

    def start(
        self,
        map_path: Path | None = None,
    ) -> None:
        if self._resources is not None:
            raise RuntimeError(
                "SLAM backend is already started"
            )

        pipeline = depthai.Pipeline()

        config = (
            spectacularAI.depthai.Configuration()
        )

        config.internalParameters = {
            "extendParameterSets": [
                "point-cloud"
            ]
        }

        if map_path is not None:
            config.mapSavePath = str(
                map_path
            )

        vio_pipeline = (
            spectacularAI.depthai.Pipeline(
                pipeline,
                config,
                self._on_mapping_output,
            )
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

            preview_queue = (
                device.getOutputQueue(
                    name="preview",
                    maxSize=1,
                    blocking=False,
                )
            )

        except Exception:
            resources.close()
            raise

        self._mapping = MappingStats()

        self._resources = resources
        self._session = session
        self._preview_queue = (
            preview_queue
        )

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

        output = (
            self._session.waitForOutput()
        )

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
            tracking_status=(
                output.status.name
            ),
            camera_to_world=(
                camera_to_world
            ),
            mapping=self._mapping,
        )

    def get_preview_frame(
        self,
    ) -> np.ndarray | None:
        if self._preview_queue is None:
            return None

        frame = (
            self._preview_queue.tryGet()
        )

        if frame is None:
            return None

        return np.array(
            frame.getCvFrame(),
            copy=True,
        )

    def _on_mapping_output(
        self,
        output,
    ) -> None:
        slam_map = output.map

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