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
        self._mapping = MappingStats()

    def start(self, map_path: Path | None = None) -> None:
        if self._resources is not None:
            raise RuntimeError("SLAM backend is already started")

        pipeline = depthai.Pipeline()
        config = spectacularAI.depthai.Configuration()
        config.internalParameters = {
            "extendParameterSets": ["point-cloud"]
        }

        if map_path is not None:
            config.mapSavePath = str(map_path)

        vio_pipeline = spectacularAI.depthai.Pipeline(
            pipeline,
            config,
            self._on_mapping_output,
        )
        resources = ExitStack()

        try:
            device = resources.enter_context(depthai.Device(pipeline))
            session = resources.enter_context(
                vio_pipeline.startSession(device)
            )
        except Exception:
            resources.close()
            raise

        self._mapping = MappingStats()
        self._resources = resources
        self._session = session

    def stop(self) -> None:
        if self._resources is None:
            return

        self._resources.close()
        self._resources = None
        self._session = None

    def wait_for_update(self) -> SlamUpdate:
        if self._session is None:
            raise RuntimeError("SLAM backend has not been started")

        output = self._session.waitForOutput()
        camera_pose = output.getCameraPose(0).pose
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

    def _on_mapping_output(self, output) -> None:
        slam_map = output.map
        dense_points = sum(
            keyframe.pointCloud.size()
            for keyframe in slam_map.keyFrames.values()
            if keyframe.pointCloud is not None
        )

        self._mapping = MappingStats(
            keyframes=len(slam_map.keyFrames),
            landmarks=len(slam_map.mapPoints),
            dense_points=dense_points,
        )