# Open3D RGB-D Odometry Prototype

This directory contains the early Project Scan RGB-D odometry prototype.

It was used to validate:

- OAK-D Pro RGB/depth synchronization
- depth alignment and metric depth
- camera intrinsics and calibration
- active IR depth performance
- IMU acquisition and timing
- IMU-to-camera coordinate transforms
- gyro-assisted RGB-D odometry
- basic translation and rotation behavior

The prototype demonstrated that gyro initialization substantially improved
Open3D RGB-D rotational convergence, but positional drift and
rotation/translation coupling remained.

Project Scan subsequently moved to an established visual-inertial SLAM backend
rather than continuing to develop custom odometry.

This code is retained for experimentation, diagnostics, and development history.
It is not part of the current production architecture.
