import cv2
import depthai as dai
import numpy as np

MIN_DEPTH_MM = 200
MAX_DEPTH_MM = 5_000
SAMPLE_RADIUS  = 3

def get_center_distance(depth_frame: np.ndarray) -> int | None:
    # return median valid depth near center of image
    h, w = depth_frame.shape
    cx, cy = w // 2, h // 2

    region = depth_frame[cy - SAMPLE_RADIUS:cy + SAMPLE_RADIUS + 1, cx - SAMPLE_RADIUS:cx + SAMPLE_RADIUS + 1]
    valid_depths = region[(region >= MIN_DEPTH_MM) & (region <= MAX_DEPTH_MM)]

    if valid_depths.size == 0:
        return None

    return int(np.median(valid_depths))

def colorize_depth(depth_frame: np.ndarray) -> np.ndarray:
    # convert mm depth map to viewable color image
    clipped = np.clip(depth_frame, MIN_DEPTH_MM, MAX_DEPTH_MM)
    normalized = cv2.normalize(clipped, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

    colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    # set invalid depth pixels to black
    colorized[depth_frame == 0] = 0
    return colorized

def main() -> None:
    pipeline = dai.Pipeline()

    left_camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right_camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    left_output = left_camera.requestFullResolutionOutput()
    right_output = right_camera.requestFullResolutionOutput()

    stereo = pipeline.create(dai.node.StereoDepth)

    left_output.link(stereo.left)
    right_output.link(stereo.right)

    stereo.setRectification(True)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(True)

    depth_queue = stereo.depth.createOutputQueue()

    with pipeline:
        pipeline.start()

        print("live depth preview. number shown is center crosshair distance mm. q to quit.")

        while pipeline.isRunning():
            depth_message = depth_queue.get()
            depth_frame = depth_message.getFrame()

            depth_view = colorize_depth(depth_frame)

            # draw crosshair
            h, w, _ = depth_view.shape
            center = (w // 2, h // 2)
            cv2.drawMarker(depth_view, center, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

            # get distance at crosshair
            distance_mm = get_center_distance(depth_frame)
            if distance_mm is None:
                distance_text = "distance: unavailable"
            else:
                distance_text = f"distance: {distance_mm} mm"
        
            cv2.putText(depth_view, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Depth Preview", depth_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()