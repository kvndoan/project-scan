import cv2
import depthai as dai

DOT_STEP = 0.1
FLOOD_STEP = 0.1
dot_intensity = 0.0
flood_intensity = 0.0

device = dai.Device()
with dai.Pipeline(device) as pipeline:
    left_camera = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_B
    )
    right_camera = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_C
    )

    left_output = left_camera.requestFullResolutionOutput(
        type=dai.ImgFrame.Type.NV12
    )
    right_output = right_camera.requestFullResolutionOutput(
        type=dai.ImgFrame.Type.NV12
    )

    left_queue = left_output.createOutputQueue()
    right_queue = right_output.createOutputQueue()

    pipeline.start()

    active_device = pipeline.getDefaultDevice()
    active_device.setIrLaserDotProjectorIntensity(dot_intensity)
    active_device.setIrFloodLightIntensity(flood_intensity)

    print("controls:")
    print("  'w'/'s': Increase/Decrease dot intensity")
    print("  'a'/'d': Increase/Decrease flood intensity")
    print("  'q': Quit")
    print()
    print(f"Current dot intensity: {dot_intensity:.1f}")
    print(f"Current flood intensity: {flood_intensity:.1f}")

    while pipeline.isRunning():
        left_frame = left_queue.get()
        right_frame = right_queue.get()

        left_image = left_frame.getCvFrame()
        right_image = right_frame.getCvFrame()

        combined_image = cv2.hconcat([left_image, right_image])
        cv2.imshow("IR Cameras", combined_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):
            dot_intensity = min(dot_intensity + DOT_STEP, 1.0)
            active_device.setIrLaserDotProjectorIntensity(dot_intensity)
            print(f"Current dot intensity: {dot_intensity:.1f}")
        elif key == ord('s'):
            dot_intensity = max(dot_intensity - DOT_STEP, 0.0)
            active_device.setIrLaserDotProjectorIntensity(dot_intensity)
            print(f"Current dot intensity: {dot_intensity:.1f}")
        elif key == ord('a'):
            flood_intensity = min(flood_intensity + FLOOD_STEP, 1.0)
            active_device.setIrFloodLightIntensity(flood_intensity)
            print(f"Current flood intensity: {flood_intensity:.1f}")
        elif key == ord('d'):
            flood_intensity = max(flood_intensity - FLOOD_STEP, 0.0)
            active_device.setIrFloodLightIntensity(flood_intensity)
            print(f"Current flood intensity: {flood_intensity:.1f}")
        
    active_device.setIrLaserDotProjectorIntensity(0.0)
    active_device.setIrFloodLightIntensity(0.0)

cv2.destroyAllWindows()