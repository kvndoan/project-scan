from datetime import timedelta

import cv2
import depthai as dai
import numpy as np

FPS = 30.0

RGB_SOCKET = dai.CameraBoardSocket.CAM_A
LEFT_SOCKET = dai.CameraBoardSocket.CAM_B
RIGHT_SOCKET = dai.CameraBoardSocket.CAM_C

RGB_SIZE = (1280, 960)
STEREO_SIZE = (640, 400)

SAMPLE_RADIUS = 3

rgb_weight = 0.6
depth_weight = 0.4

def get_center_distance(depth_frame: np.ndarray) -> int | None:
    # return median valid depth near center of image
    h, w = depth_frame.shape
    cx, cy = w // 2, h // 2

    region = depth_frame[cy - SAMPLE_RADIUS:cy + SAMPLE_RADIUS + 1, cx - SAMPLE_RADIUS:cx + SAMPLE_RADIUS + 1]
    valid_depths = region[region > 0]

    if valid_depths.size == 0:
        return None

    return int(np.median(valid_depths))

def colorize_depth(depth_frame: np.ndarray) -> np.ndarray:
    # convert mm depth map to viewable color image
    invalid_mask = depth_frame == 0
    valid_depths = depth_frame[~invalid_mask]

    if valid_depths.size == 0:
        return np.zeros((depth_frame.shape[0], depth_frame.shape[1], 3), dtype=np.uint8)
    
    min_depth = np.percentile(valid_depths, 3)
    max_depth = np.percentile(valid_depths, 95)

    log_depth = np.zeros_like(depth_frame, dtype=np.float32)

    np.log(depth_frame, where=depth_frame != 0, out=log_depth)
    log_min_depth = np.log(min_depth)
    log_max_depth = np.log(max_depth)

    log_depth = np.clip(log_depth, log_min_depth, log_max_depth)
    normalized = np.interp(log_depth, (log_min_depth, log_max_depth), (0, 255)).astype(np.uint8)

    colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colorized[invalid_mask] = 0

    return colorized

def update_blend_weights(percent_rgb: int) -> None:
    # update rgb and depth blend percentages from slider
    global rgb_weight
    global depth_weight

    rgb_weight = percent_rgb / 100.0
    depth_weight = 1.0 - rgb_weight

def draw_center_measurement(image: np.ndarray, depth_frame: np.ndarray) -> None:
    # draw crosshair and center distance measurement on image
    h, w, _ = image.shape
    center = (w // 2, h // 2)

    cv2.drawMarker(image, center, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

    distance_mm = get_center_distance(depth_frame)
    if distance_mm is not None:
        text = f"{distance_mm} mm"
    else:
        text = "N/A"
        
    cv2.putText(image, text, (center[0] + 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

def main() -> None:
    pipeline = dai.Pipeline()

    rgb_camera = pipeline.create(dai.node.Camera).build(RGB_SOCKET)
    left_camera = pipeline.create(dai.node.Camera).build(LEFT_SOCKET)
    right_camera = pipeline.create(dai.node.Camera).build(RIGHT_SOCKET)

    stereo = pipeline.create(dai.node.StereoDepth)
    sync = pipeline.create(dai.node.Sync)

    rgb_output = rgb_camera.requestOutput(size=RGB_SIZE, fps=FPS, enableUndistortion=True)
    left_output = left_camera.requestOutput(size=STEREO_SIZE, fps=FPS)
    right_output = right_camera.requestOutput(size=STEREO_SIZE, fps=FPS)

    left_output.link(stereo.left)
    right_output.link(stereo.right)

    stereo.setExtendedDisparity(True)
    stereo.setLeftRightCheck(True)

    rgb_output.link(stereo.inputAlignTo)
    
    rgb_output.link(sync.inputs["rgb"])
    stereo.depth.link(sync.inputs["depth"])

    sync.setSyncThreshold(timedelta(seconds=1 / (2 * FPS)))

    output_queue = sync.out.createOutputQueue(maxSize=1, blocking=False)

    window_name = "RGB + Depth Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    cv2.createTrackbar("RGB %", window_name, int(rgb_weight * 100), 100, update_blend_weights)

    with pipeline:
        pipeline.start()

        print("rgb + depth preview started. user slider to change rgb/depth blend. press 'q' to quit.")

        while pipeline.isRunning():
            message_groups = output_queue.tryGetAll()

            if not message_groups:
                cv2.waitKey(1)
                continue

            message_group = message_groups[-1]

            rgb_frame = message_group["rgb"].getCvFrame()
            depth_frame = message_group["depth"].getFrame()

            depth_view = colorize_depth(depth_frame)

            # safety check for alignment, protects against unexpected output-size differences
            if depth_view.shape[:2] != rgb_frame.shape[:2]:
                depth_view = cv2.resize(depth_view, (rgb_frame.shape[1], rgb_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                depth_frame = cv2.resize(depth_frame, (rgb_frame.shape[1], rgb_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            blended_view = cv2.addWeighted(rgb_frame, rgb_weight, depth_view, depth_weight, 0)
            draw_center_measurement(blended_view, depth_frame)

            cv2.imshow("RGB", rgb_frame)
            cv2.imshow("Depth", depth_view)
            cv2.imshow(window_name, blended_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()