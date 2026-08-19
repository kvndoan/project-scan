from collections import deque
from dataclasses import dataclass
from datetime import timedelta

import depthai as dai
import numpy as np

from .types import CameraIntrinsics, IMUSample, SensorFrame


@dataclass(frozen=True)
class OakDSensorConfig:
    fps: float = 30.0

    rgb_size: tuple[int, int] = (1280, 960)
    stereo_size: tuple[int, int] = (640, 400)

    accel_rate_hz: int = 480
    gyro_rate_hz: int = 400

    # OAK-D Pro IR dot projector intensity.
    #
    # 0.0 = off
    # 1.0 = maximum
    ir_dot_intensity: float = 1.0


class OakDSensor:
    """
    Owns all DepthAI-specific RGB, stereo-depth, IMU, and timing logic.

    IMU acceleration and angular velocity are converted from the
    physical IMU coordinate frame into the RGB camera coordinate frame
    before SensorFrame objects are exposed to downstream code.
    """

    RGB_SOCKET = dai.CameraBoardSocket.CAM_A
    LEFT_SOCKET = dai.CameraBoardSocket.CAM_B
    RIGHT_SOCKET = dai.CameraBoardSocket.CAM_C

    def __init__(
        self,
        config: OakDSensorConfig | None = None,
    ) -> None:
        self.config = config or OakDSensorConfig()

        if not 0.0 <= self.config.ir_dot_intensity <= 1.0:
            raise ValueError(
                "ir_dot_intensity must be between 0.0 and 1.0"
            )

        self._pipeline: dai.Pipeline | None = None

        self._rgbd_queue = None
        self._imu_queue = None

        self._pending_imu: deque[IMUSample] = deque()

        self._intrinsics: CameraIntrinsics | None = None

        # Rotation mapping vectors from the physical IMU coordinate
        # system into CAM_A's optical coordinate system.
        self._R_imu_to_camera: np.ndarray | None = None

    @property
    def intrinsics(self) -> CameraIntrinsics:
        if self._intrinsics is None:
            raise RuntimeError(
                "Sensor has not been started yet"
            )

        return self._intrinsics

    def __enter__(self) -> "OakDSensor":
        self.start()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()

    def start(self) -> None:
        if self._pipeline is not None:
            raise RuntimeError(
                "Sensor is already started"
            )

        pipeline = dai.Pipeline()

        # ------------------------------------------------------------
        # Device calibration
        # ------------------------------------------------------------

        device = pipeline.getDefaultDevice()
        calibration = device.readCalibration()

        # ------------------------------------------------------------
        # RGB calibration focus
        # ------------------------------------------------------------

        rgb_lens_position = (
            calibration.getLensPosition(
                self.RGB_SOCKET
            )
        )

        if not rgb_lens_position:
            raise RuntimeError(
                "No calibrated RGB lens position was found for CAM_A"
            )

        # ------------------------------------------------------------
        # IMU -> RGB camera extrinsics
        # ------------------------------------------------------------
        #
        # IMU measurements are physically reported in the IMU's own
        # coordinate frame.
        #
        # Downstream pose estimation works in the RGB-camera frame, so
        # use the device's factory calibration to rotate IMU vectors
        # into CAM_A coordinates.
        #
        # Translation is irrelevant for angular velocity and linear
        # acceleration vectors, so we only keep the 3x3 rotation.
        # ------------------------------------------------------------

        imu_to_camera = np.asarray(
            calibration.getImuToCameraExtrinsics(
                self.RGB_SOCKET,
                False,
            ),
            dtype=np.float64,
        )

        if imu_to_camera.shape != (4, 4):
            raise RuntimeError(
                "Expected 4x4 IMU-to-camera extrinsic matrix, "
                f"got {imu_to_camera.shape}"
            )

        self._R_imu_to_camera = (
            imu_to_camera[
                :3,
                :3,
            ]
        )

        # ------------------------------------------------------------
        # Cameras
        # ------------------------------------------------------------

        rgb_camera = (
            pipeline
            .create(dai.node.Camera)
            .build(
                self.RGB_SOCKET
            )
        )

        left_camera = (
            pipeline
            .create(dai.node.Camera)
            .build(
                self.LEFT_SOCKET
            )
        )

        right_camera = (
            pipeline
            .create(dai.node.Camera)
            .build(
                self.RIGHT_SOCKET
            )
        )

        # Keep RGB optics fixed at the lens position used during
        # factory calibration.
        rgb_camera.initialControl.setManualFocus(
            int(rgb_lens_position)
        )

        rgb_output = rgb_camera.requestOutput(
            size=self.config.rgb_size,
            fps=self.config.fps,
            enableUndistortion=True,
        )

        left_output = left_camera.requestOutput(
            size=self.config.stereo_size,
            fps=self.config.fps,
        )

        right_output = right_camera.requestOutput(
            size=self.config.stereo_size,
            fps=self.config.fps,
        )

        # ------------------------------------------------------------
        # Stereo depth
        # ------------------------------------------------------------

        stereo = pipeline.create(
            dai.node.StereoDepth
        )

        left_output.link(
            stereo.left
        )

        right_output.link(
            stereo.right
        )

        stereo.setExtendedDisparity(
            True
        )

        stereo.setLeftRightCheck(
            True
        )

        # Align depth pixels with the RGB camera image.
        rgb_output.link(
            stereo.inputAlignTo
        )

        # ------------------------------------------------------------
        # RGB + depth synchronization
        # ------------------------------------------------------------

        sync = pipeline.create(
            dai.node.Sync
        )

        rgb_output.link(
            sync.inputs["rgb"]
        )

        stereo.depth.link(
            sync.inputs["depth"]
        )

        sync.setSyncThreshold(
            timedelta(
                seconds=(
                    1
                    / (
                        2
                        * self.config.fps
                    )
                )
            )
        )

        # ------------------------------------------------------------
        # IMU
        # ------------------------------------------------------------

        imu = pipeline.create(
            dai.node.IMU
        )

        imu.enableIMUSensor(
            dai.IMUSensor.ACCELEROMETER_UNCALIBRATED,
            self.config.accel_rate_hz,
        )

        imu.enableIMUSensor(
            dai.IMUSensor.GYROSCOPE_UNCALIBRATED,
            self.config.gyro_rate_hz,
        )

        imu.setBatchReportThreshold(
            1
        )

        imu.setMaxBatchReports(
            10
        )

        # ------------------------------------------------------------
        # Host output queues
        # ------------------------------------------------------------

        self._rgbd_queue = (
            sync.out.createOutputQueue(
                maxSize=1,
                blocking=False,
            )
        )

        self._imu_queue = (
            imu.out.createOutputQueue(
                maxSize=50,
                blocking=False,
            )
        )

        # ------------------------------------------------------------
        # Start device
        # ------------------------------------------------------------

        pipeline.start()

        # ------------------------------------------------------------
        # Enable active stereo
        # ------------------------------------------------------------

        device.setIrLaserDotProjectorIntensity(
            self.config.ir_dot_intensity
        )

        print(
            "OAK-D RGB focus locked to calibration lens position: "
            f"{rgb_lens_position}"
        )

        print(
            "OAK-D IR dot projector intensity: "
            f"{self.config.ir_dot_intensity:.2f}"
        )

        print(
            "OAK-D IMU vectors transformed into CAM_A coordinates."
        )

        print(
            "R_imu_to_camera:"
        )

        print(
            self._R_imu_to_camera
        )

        # ------------------------------------------------------------
        # RGB intrinsics
        # ------------------------------------------------------------

        width, height = (
            self.config.rgb_size
        )

        intrinsic_matrix = np.asarray(
            calibration.getCameraIntrinsics(
                self.RGB_SOCKET,
                width,
                height,
            ),
            dtype=np.float64,
        )

        self._intrinsics = (
            CameraIntrinsics(
                width=width,
                height=height,
                fx=float(
                    intrinsic_matrix[
                        0,
                        0,
                    ]
                ),
                fy=float(
                    intrinsic_matrix[
                        1,
                        1,
                    ]
                ),
                cx=float(
                    intrinsic_matrix[
                        0,
                        2,
                    ]
                ),
                cy=float(
                    intrinsic_matrix[
                        1,
                        2,
                    ]
                ),
            )
        )

        self._pipeline = pipeline

    def close(self) -> None:
        if (
            self._pipeline is not None
            and self._pipeline.isRunning()
        ):
            self._pipeline.stop()

        self._pipeline = None

        self._rgbd_queue = None
        self._imu_queue = None

        self._pending_imu.clear()

        self._intrinsics = None
        self._R_imu_to_camera = None

    def is_running(self) -> bool:
        return (
            self._pipeline is not None
            and self._pipeline.isRunning()
        )

    def poll_frame(
        self,
    ) -> SensorFrame | None:
        """
        Return the newest synchronized RGB-D frame.

        Returns None if no complete RGB-D observation is currently ready.
        """

        if (
            self._pipeline is None
            or self._rgbd_queue is None
            or self._imu_queue is None
        ):
            raise RuntimeError(
                "Sensor is not started"
            )

        # Collect any IMU measurements that have reached the host.
        self._drain_imu_queue()

        # Get all currently available synchronized RGB-D groups.
        message_groups = (
            self._rgbd_queue.tryGetAll()
        )

        if not message_groups:
            return None

        # Low latency matters more than processing every preview frame.
        message_group = (
            message_groups[-1]
        )

        rgb_message = (
            message_group["rgb"]
        )

        depth_message = (
            message_group["depth"]
        )

        rgb = (
            rgb_message.getCvFrame()
        )

        depth_mm = (
            depth_message.getFrame()
        )

        # Mapping requires aligned RGB and depth.
        if (
            depth_mm.shape
            != rgb.shape[:2]
        ):
            raise RuntimeError(
                "Aligned depth and RGB shapes differ: "
                f"depth={depth_mm.shape}, "
                f"rgb={rgb.shape[:2]}"
            )

        # Keep camera and IMU synchronization comparisons on the
        # OAK-D device clock.
        frame_timestamp_s = (
            rgb_message
            .getTimestampDevice()
            .total_seconds()
        )

        imu_samples = (
            self._take_imu_through(
                frame_timestamp_s
            )
        )

        return SensorFrame(
            sequence_num=(
                rgb_message
                .getSequenceNum()
            ),
            timestamp_s=frame_timestamp_s,
            rgb=rgb,
            depth_mm=depth_mm,
            imu_samples=imu_samples,
        )

    def _drain_imu_queue(
        self,
    ) -> None:
        """
        Move all currently available IMU reports into our local buffer.

        IMU vectors are converted into CAM_A coordinates before being
        exposed to downstream code.
        """

        if self._R_imu_to_camera is None:
            raise RuntimeError(
                "IMU-to-camera calibration is not initialized"
            )

        for imu_data in (
            self._imu_queue.tryGetAll()
        ):
            for packet in (
                imu_data.packets
            ):
                accel = (
                    packet.acceleroMeter
                )

                gyro = (
                    packet.gyroscope
                )

                accel_imu = np.array(
                    [
                        accel.x,
                        accel.y,
                        accel.z,
                    ],
                    dtype=np.float64,
                )

                gyro_imu = np.array(
                    [
                        gyro.x,
                        gyro.y,
                        gyro.z,
                    ],
                    dtype=np.float64,
                )

                # Rotate vector quantities from the physical IMU frame
                # into the RGB camera frame.
                accel_camera = (
                    self._R_imu_to_camera
                    @ accel_imu
                )

                gyro_camera = (
                    self._R_imu_to_camera
                    @ gyro_imu
                )

                sample = IMUSample(
                    accel_timestamp_s=(
                        accel
                        .getTimestampDevice()
                        .total_seconds()
                    ),
                    gyro_timestamp_s=(
                        gyro
                        .getTimestampDevice()
                        .total_seconds()
                    ),
                    accel_mps2=(
                        accel_camera
                    ),
                    gyro_rps=(
                        gyro_camera
                    ),
                )

                self._pending_imu.append(
                    sample
                )

    def _take_imu_through(
        self,
        frame_timestamp_s: float,
    ) -> tuple[IMUSample, ...]:
        """
        Remove and return IMU samples whose timestamps are no later
        than the current camera frame.
        """

        samples: list[
            IMUSample
        ] = []

        while (
            self._pending_imu
            and (
                self
                ._pending_imu[0]
                .timestamp_s
                <= frame_timestamp_s
            )
        ):
            samples.append(
                self._pending_imu.popleft()
            )

        return tuple(
            samples
        )