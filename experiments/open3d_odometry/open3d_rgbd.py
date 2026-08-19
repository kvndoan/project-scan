from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d

from project_scan.sensors import CameraIntrinsics, SensorFrame

from .base import PoseEstimator
from .types import PoseEstimate


@dataclass(frozen=True)
class Open3DRGBDOdometryConfig:
    """
    Configuration for the Open3D RGB-D odometry backend.
    """

    # 1280x960 input becomes 320x240 for odometry.
    processing_scale: float = 0.25

    # OAK-D depth is stored in millimeters.
    depth_scale: float = 1000.0

    # Ignore geometry farther than this distance.
    depth_trunc_m: float = 4.0

    # Use camera-frame gyro measurements to provide Open3D with an
    # initial rotational motion estimate between RGB-D frames.
    use_imu_rotation_prior: bool = True

    # Ignore an individual IMU integration interval if its timestamps
    # indicate a suspiciously large gap.
    max_imu_interval_s: float = 0.05

    # Temporary MVP sanity limits.
    max_relative_translation_m: float = 0.35
    max_relative_rotation_deg: float = 35.0


@dataclass(frozen=True)
class Open3DOdometryDiagnostics:
    """
    Debug information describing the most recent odometry step.
    """

    frame_gap: int
    dt_s: float

    relative_translation_m: float
    relative_rotation_deg: float

    relative_translation_xyz_m: tuple[
        float,
        float,
        float,
    ]

    # Gyro-derived rotation used to initialize Open3D for this step.
    imu_prior_rotation_deg: float
    imu_samples_used: int

    # Difference between the gyro rotational prior and Open3D's final
    # rotational estimate for this frame-to-frame step.
    imu_open3d_rotation_error_deg: float

    open3d_success: bool
    sanity_ok: bool

    rejection_reason: str | None


class Open3DRGBDOdometry(PoseEstimator):
    """
    Estimates camera motion from consecutive synchronized RGB-D frames.

    The OAK-D gyro is used only as a short-term rotational INITIAL GUESS
    for Open3D. Open3D still estimates the final full 6-DoF transform.
    """

    def __init__(
        self,
        config: Open3DRGBDOdometryConfig | None = None,
    ) -> None:
        self.config = config or Open3DRGBDOdometryConfig()

        if not 0.0 < self.config.processing_scale <= 1.0:
            raise ValueError(
                "processing_scale must be greater than 0 and at most 1"
            )

        if self.config.max_imu_interval_s <= 0.0:
            raise ValueError(
                "max_imu_interval_s must be greater than 0"
            )

        self._intrinsics: (
            o3d.camera.PinholeCameraIntrinsic | None
        ) = None

        self._previous_rgbd: (
            o3d.geometry.RGBDImage | None
        ) = None

        self._previous_timestamp_s: float | None = None
        self._previous_sequence_num: int | None = None

        # Persistent camera pose:
        #
        # camera coordinates
        #       ↓
        # T_world_camera
        #       ↓
        # world coordinates
        self._T_world_camera = np.eye(
            4,
            dtype=np.float64,
        )

        self._odometry_option = (
            o3d.pipelines.odometry.OdometryOption()
        )

        self._odometry_option.depth_max = (
            self.config.depth_trunc_m
        )

        self._last_diagnostics = (
            self._empty_diagnostics()
        )

    @property
    def last_diagnostics(
        self,
    ) -> Open3DOdometryDiagnostics:
        """
        Return debugging information from the latest update.
        """

        return self._last_diagnostics

    def reset(
        self,
        intrinsics: CameraIntrinsics,
    ) -> None:
        """
        Reset tracking and establish a new world origin.
        """

        scale = self.config.processing_scale

        width = round(
            intrinsics.width * scale
        )

        height = round(
            intrinsics.height * scale
        )

        scale_x = (
            width
            / intrinsics.width
        )

        scale_y = (
            height
            / intrinsics.height
        )

        self._intrinsics = (
            o3d.camera.PinholeCameraIntrinsic(
                width,
                height,
                intrinsics.fx * scale_x,
                intrinsics.fy * scale_y,
                intrinsics.cx * scale_x,
                intrinsics.cy * scale_y,
            )
        )

        self._previous_rgbd = None
        self._previous_timestamp_s = None
        self._previous_sequence_num = None

        self._T_world_camera = np.eye(
            4,
            dtype=np.float64,
        )

        self._last_diagnostics = (
            self._empty_diagnostics()
        )

    def update(
        self,
        frame: SensorFrame,
    ) -> PoseEstimate:
        """
        Estimate the world-space camera pose for one sensor frame.
        """

        if self._intrinsics is None:
            raise RuntimeError(
                "Pose estimator must be reset with camera intrinsics "
                "before calling update()"
            )

        current_rgbd = self._make_rgbd(
            frame
        )

        # ------------------------------------------------------------
        # First frame
        # ------------------------------------------------------------

        if self._previous_rgbd is None:
            self._set_reference_frame(
                rgbd=current_rgbd,
                frame=frame,
            )

            self._last_diagnostics = (
                self._empty_diagnostics()
            )

            return PoseEstimate(
                timestamp_s=frame.timestamp_s,
                T_world_camera=(
                    self._T_world_camera.copy()
                ),
                tracking_ok=True,
            )

        # ------------------------------------------------------------
        # Timing/frame diagnostics
        # ------------------------------------------------------------

        dt_s = (
            frame.timestamp_s
            - self._previous_timestamp_s
        )

        frame_gap = (
            frame.sequence_num
            - self._previous_sequence_num
        )

        # ------------------------------------------------------------
        # IMU rotational prior
        # ------------------------------------------------------------
        #
        # Gyro measurements are already expressed in CAM_A coordinates
        # by OakDSensor.
        #
        # Integrating body-frame angular velocity produces:
        #
        # current camera coordinates
        #       ↓
        # R_previous_current
        #       ↓
        # previous camera coordinates
        #
        # Open3D's source is PREVIOUS and target is CURRENT, so its
        # initial source->target transform requires the inverse:
        #
        # R_current_previous = R_previous_current.T
        # ------------------------------------------------------------

        (
            odometry_init,
            imu_samples_used,
            imu_prior_rotation_deg,
        ) = self._make_odometry_init_from_imu(
            frame
        )

        # ------------------------------------------------------------
        # Open3D relative odometry
        # ------------------------------------------------------------

        (
            success,
            T_current_previous,
            _,
        ) = (
            o3d.pipelines.odometry
            .compute_rgbd_odometry(
                self._previous_rgbd,
                current_rgbd,
                self._intrinsics,
                odometry_init,
                (
                    o3d.pipelines.odometry
                    .RGBDOdometryJacobianFromHybridTerm()
                ),
                self._odometry_option,
            )
        )

        # ------------------------------------------------------------
        # Validate candidate transform
        # ------------------------------------------------------------

        relative_translation_m = 0.0
        relative_rotation_deg = 0.0

        relative_translation_xyz_m = (
            0.0,
            0.0,
            0.0,
        )

        imu_open3d_rotation_error_deg = 0.0

        sanity_ok = False
        rejection_reason: str | None = None

        T_previous_current: np.ndarray | None = None

        if success:
            # Open3D returned:
            #
            # previous coordinates
            #       ↓
            # T_current_previous
            #       ↓
            # current coordinates
            #
            # Camera pose accumulation requires the reverse direction.
            T_previous_current = np.linalg.inv(
                T_current_previous
            )

            relative_translation = (
                T_previous_current[
                    :3,
                    3,
                ]
            )

            relative_translation_xyz_m = (
                float(
                    relative_translation[0]
                ),
                float(
                    relative_translation[1]
                ),
                float(
                    relative_translation[2]
                ),
            )

            relative_translation_m = float(
                np.linalg.norm(
                    relative_translation
                )
            )

            relative_rotation_deg = (
                self._rotation_magnitude_deg(
                    T_previous_current
                )
            )

            # Both odometry_init and T_current_previous are expressed
            # as previous/source -> current/target transforms.
            #
            # Compare only their rotation components.
            R_init = (
                odometry_init[
                    :3,
                    :3,
                ]
            )

            R_open3d = (
                T_current_previous[
                    :3,
                    :3,
                ]
            )

            R_init_to_open3d = (
                R_init.T
                @ R_open3d
            )

            imu_open3d_rotation_error_deg = (
                self._rotation_matrix_magnitude_deg(
                    R_init_to_open3d
                )
            )

            translation_ok = (
                relative_translation_m
                <= (
                    self.config
                    .max_relative_translation_m
                )
            )

            rotation_ok = (
                relative_rotation_deg
                <= (
                    self.config
                    .max_relative_rotation_deg
                )
            )

            sanity_ok = (
                translation_ok
                and rotation_ok
            )

            if not sanity_ok:
                reasons: list[str] = []

                if not translation_ok:
                    reasons.append(
                        "translation"
                    )

                if not rotation_ok:
                    reasons.append(
                        "rotation"
                    )

                rejection_reason = (
                    "+".join(
                        reasons
                    )
                )

        else:
            rejection_reason = (
                "open3d_failed"
            )

        # ------------------------------------------------------------
        # Accept or reject the relative motion
        # ------------------------------------------------------------

        tracking_ok = bool(
            success
            and sanity_ok
        )

        if tracking_ok:
            assert (
                T_previous_current
                is not None
            )

            self._T_world_camera = (
                self._T_world_camera
                @ T_previous_current
            )

        # Always advance the local RGB-D reference.
        self._set_reference_frame(
            rgbd=current_rgbd,
            frame=frame,
        )

        self._last_diagnostics = (
            Open3DOdometryDiagnostics(
                frame_gap=frame_gap,
                dt_s=dt_s,
                relative_translation_m=(
                    relative_translation_m
                ),
                relative_rotation_deg=(
                    relative_rotation_deg
                ),
                relative_translation_xyz_m=(
                    relative_translation_xyz_m
                ),
                imu_prior_rotation_deg=(
                    imu_prior_rotation_deg
                ),
                imu_samples_used=(
                    imu_samples_used
                ),
                imu_open3d_rotation_error_deg=(
                    imu_open3d_rotation_error_deg
                ),
                open3d_success=bool(
                    success
                ),
                sanity_ok=sanity_ok,
                rejection_reason=(
                    rejection_reason
                ),
            )
        )

        return PoseEstimate(
            timestamp_s=frame.timestamp_s,
            T_world_camera=(
                self._T_world_camera.copy()
            ),
            tracking_ok=tracking_ok,
        )

    def _make_odometry_init_from_imu(
        self,
        frame: SensorFrame,
    ) -> tuple[
        np.ndarray,
        int,
        float,
    ]:
        """
        Build Open3D's initial source->target transform from the gyro.

        Only rotation is predicted. Translation remains zero.
        """

        odometry_init = np.eye(
            4,
            dtype=np.float64,
        )

        if (
            not self.config.use_imu_rotation_prior
            or self._previous_timestamp_s is None
        ):
            return (
                odometry_init,
                0,
                0.0,
            )

        previous_timestamp_s = (
            self._previous_timestamp_s
        )

        current_timestamp_s = (
            frame.timestamp_s
        )

        # Only use gyro samples that belong to the exact interval
        # between the two RGB-D observations being compared.
        gyro_samples = [
            sample
            for sample in frame.imu_samples
            if (
                previous_timestamp_s
                < sample.gyro_timestamp_s
                <= current_timestamp_s
            )
        ]

        if not gyro_samples:
            return (
                odometry_init,
                0,
                0.0,
            )

        # Maps CURRENT camera coordinates into PREVIOUS camera
        # coordinates.
        R_previous_current = np.eye(
            3,
            dtype=np.float64,
        )

        integration_timestamp_s = (
            previous_timestamp_s
        )

        last_gyro_rps: (
            np.ndarray | None
        ) = None

        samples_used = 0

        for sample in gyro_samples:
            sample_timestamp_s = (
                sample.gyro_timestamp_s
            )

            dt_s = (
                sample_timestamp_s
                - integration_timestamp_s
            )

            if dt_s <= 0.0:
                continue

            if (
                dt_s
                > self.config.max_imu_interval_s
            ):
                # We appear to be missing too much IMU history for
                # this interval. Re-anchor time here rather than
                # integrating one measurement across a huge gap.
                integration_timestamp_s = (
                    sample_timestamp_s
                )

                last_gyro_rps = (
                    sample.gyro_rps
                )

                continue

            delta_rotation_vector = (
                sample.gyro_rps
                * dt_s
            )

            delta_rotation, _ = (
                cv2.Rodrigues(
                    delta_rotation_vector
                )
            )

            R_previous_current = (
                R_previous_current
                @ delta_rotation
            )

            integration_timestamp_s = (
                sample_timestamp_s
            )

            last_gyro_rps = (
                sample.gyro_rps
            )

            samples_used += 1

        # Usually the final gyro timestamp is a couple milliseconds
        # before the camera timestamp. Cover that tiny remaining tail
        # using the most recent angular-rate measurement.
        if last_gyro_rps is not None:
            tail_dt_s = (
                current_timestamp_s
                - integration_timestamp_s
            )

            if (
                0.0
                < tail_dt_s
                <= self.config.max_imu_interval_s
            ):
                tail_rotation_vector = (
                    last_gyro_rps
                    * tail_dt_s
                )

                tail_rotation, _ = (
                    cv2.Rodrigues(
                        tail_rotation_vector
                    )
                )

                R_previous_current = (
                    R_previous_current
                    @ tail_rotation
                )

        # Open3D needs PREVIOUS(source) -> CURRENT(target).
        R_current_previous = (
            R_previous_current.T
        )

        odometry_init[
            :3,
            :3,
        ] = R_current_previous

        imu_prior_rotation_deg = (
            self._rotation_matrix_magnitude_deg(
                R_current_previous
            )
        )

        return (
            odometry_init,
            samples_used,
            imu_prior_rotation_deg,
        )

    def _set_reference_frame(
        self,
        rgbd: o3d.geometry.RGBDImage,
        frame: SensorFrame,
    ) -> None:
        """
        Make this observation the reference used by the next update.
        """

        self._previous_rgbd = rgbd

        self._previous_timestamp_s = (
            frame.timestamp_s
        )

        self._previous_sequence_num = (
            frame.sequence_num
        )

    def _make_rgbd(
        self,
        frame: SensorFrame,
    ) -> o3d.geometry.RGBDImage:
        """
        Convert Project Scan sensor data into Open3D RGB-D data.
        """

        rgb = cv2.cvtColor(
            frame.rgb,
            cv2.COLOR_BGR2RGB,
        )

        depth_mm = (
            frame.depth_mm
        )

        scale = (
            self.config.processing_scale
        )

        if scale != 1.0:
            width = round(
                frame.rgb.shape[1]
                * scale
            )

            height = round(
                frame.rgb.shape[0]
                * scale
            )

            rgb = cv2.resize(
                rgb,
                (
                    width,
                    height,
                ),
                interpolation=(
                    cv2.INTER_AREA
                ),
            )

            depth_mm = cv2.resize(
                depth_mm,
                (
                    width,
                    height,
                ),
                interpolation=(
                    cv2.INTER_NEAREST
                ),
            )

        rgb = np.ascontiguousarray(
            rgb
        )

        depth_mm = np.ascontiguousarray(
            depth_mm,
            dtype=np.uint16,
        )

        color_image = (
            o3d.geometry.Image(
                rgb
            )
        )

        depth_image = (
            o3d.geometry.Image(
                depth_mm
            )
        )

        return (
            o3d.geometry.RGBDImage
            .create_from_color_and_depth(
                color_image,
                depth_image,
                depth_scale=(
                    self.config.depth_scale
                ),
                depth_trunc=(
                    self.config.depth_trunc_m
                ),
                convert_rgb_to_intensity=True,
            )
        )

    def _empty_diagnostics(
        self,
    ) -> Open3DOdometryDiagnostics:
        return (
            Open3DOdometryDiagnostics(
                frame_gap=0,
                dt_s=0.0,
                relative_translation_m=0.0,
                relative_rotation_deg=0.0,
                relative_translation_xyz_m=(
                    0.0,
                    0.0,
                    0.0,
                ),
                imu_prior_rotation_deg=0.0,
                imu_samples_used=0,
                imu_open3d_rotation_error_deg=0.0,
                open3d_success=True,
                sanity_ok=True,
                rejection_reason=None,
            )
        )

    @classmethod
    def _rotation_magnitude_deg(
        cls,
        transform: np.ndarray,
    ) -> float:
        """
        Return a transform's rotation magnitude in degrees.
        """

        return (
            cls._rotation_matrix_magnitude_deg(
                transform[
                    :3,
                    :3,
                ]
            )
        )

    @staticmethod
    def _rotation_matrix_magnitude_deg(
        rotation: np.ndarray,
    ) -> float:
        """
        Return how far a rotation matrix is rotated from identity.
        """

        cos_angle = (
            np.trace(
                rotation
            )
            - 1.0
        ) / 2.0

        cos_angle = np.clip(
            cos_angle,
            -1.0,
            1.0,
        )

        return float(
            np.degrees(
                np.arccos(
                    cos_angle
                )
            )
        )