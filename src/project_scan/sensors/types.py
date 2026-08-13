from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IMUSample:
    """One paired accelerometer/gyroscope report from the OAK-D IMU."""

    accel_timestamp_s: float
    gyro_timestamp_s: float

    accel_mps2: np.ndarray
    gyro_rps: np.ndarray

    @property
    def timestamp_s(self) -> float:
        """
        Return a timestamp representing this paired IMU sample.

        Accelerometer and gyroscope reports can have slightly different
        timestamps, so use the newer timestamp when associating this sample
        with a camera frame.
        """
        return max(
            self.accel_timestamp_s,
            self.gyro_timestamp_s,
        )


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole-camera parameters for the RGB image used by mapping."""

    width: int
    height: int

    fx: float
    fy: float
    cx: float
    cy: float

    def as_matrix(self) -> np.ndarray:
        """
        Return the standard 3x3 camera intrinsic matrix.
        """
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class SensorFrame:
    """
    One synchronized RGB-D observation and the IMU data associated with it.
    """

    sequence_num: int
    timestamp_s: float

    rgb: np.ndarray
    depth_mm: np.ndarray

    imu_samples: tuple[IMUSample, ...]