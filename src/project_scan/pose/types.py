from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoseEstimate:
    """
    Estimated 6-DoF camera pose for one sensor observation.
    """

    timestamp_s: float

    # 4x4 rigid transform mapping camera-frame coordinates
    # into the persistent world coordinate frame.
    T_world_camera: np.ndarray

    tracking_ok: bool