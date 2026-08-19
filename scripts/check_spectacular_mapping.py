import threading

import depthai
import spectacularAI

from spectacularAI.cli.visualization.visualizer import (
    Visualizer,
    VisualizerArgs,
)


def main() -> None:
    print("Project Scan Spectacular AI mapping check")
    print()

    vis_args = VisualizerArgs()
    vis_args.targetFps = 30

    visualizer = Visualizer(vis_args)

    def on_mapping_output(output) -> None:
        visualizer.onMappingOutput(output)

        keyframes = len(
            output.map.keyFrames
        )

        map_points = len(
            output.map.mapPoints
        )

        dense_points = sum(
            keyframe.pointCloud.size()
            for keyframe in output.map.keyFrames.values()
            if keyframe.pointCloud is not None
        )

        print(
            "\r"
            f"keyframes={keyframes:<4} "
            f"landmarks={map_points:<6} "
            f"dense_points={dense_points:<8}",
            end="",
            flush=True,
        )

    def capture_loop() -> None:
        pipeline = depthai.Pipeline()

        config = spectacularAI.depthai.Configuration()

        config.internalParameters = {
            "extendParameterSets": [
                "point-cloud"
            ]
        }

        vio_pipeline = spectacularAI.depthai.Pipeline(
            pipeline,
            config,
            on_mapping_output,
        )

        print("Starting OAK-D Pro...")

        with depthai.Device(pipeline) as device, (
            vio_pipeline.startSession(device)
        ) as session:
            print("Mapping started.")
            print("Move the camera through the room.")
            print("Close the visualization window to stop.")
            print()

            while not visualizer.shouldQuit:
                output = session.waitForOutput()

                visualizer.onVioOutput(
                    output.getCameraPose(0),
                    status=output.status,
                )

    thread = threading.Thread(
        target=capture_loop
    )

    thread.start()

    visualizer.run()

    thread.join()

    print()
    print("Mapping check stopped.")


if __name__ == "__main__":
    main()