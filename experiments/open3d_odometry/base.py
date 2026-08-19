from abc import ABC, abstractmethod

from project_scan.sensors import CameraIntrinsics, SensorFrame

from .types import PoseEstimate


class PoseEstimator(ABC):
    """
    Interface implemented by Project Scan pose-estimation backends.
    """

    @abstractmethod
    def reset(
        self,
        intrinsics: CameraIntrinsics,
    ) -> None:
        """
        Reset tracking and establish a new world origin.
        """
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        frame: SensorFrame,
    ) -> PoseEstimate:
        """
        Process one sensor observation and estimate its camera pose.
        """
        raise NotImplementedError