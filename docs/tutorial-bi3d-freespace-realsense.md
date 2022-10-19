# Tutorial for Freespace Segmentation using a RealSense Camera

<div align="center"><img src="../resources/realsense_example.gif"  width="600px"/></div>

## Overview

This tutorial demonstrates how to use a [RealSense](https://www.intel.com/content/www/us/en/architecture-and-technology/realsense-overview.html) camera and [isaac_ros_bi3d_freespace](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_proximity_segmentation) to create a local occupancy grid.

> Note: This tutorial has been tested with a RealSense D455/D435 connected to an x86 PC with an NVIDIA graphics card, as well as a Jetson Xavier AGX.

## Tutorial Walkthrough

1. Complete the [RealSense setup tutorial](https://github.com/NVIDIA-ISAAC-ROS/.github/blob/main/profile/realsense-setup.md).
2. Complete steps 1-7 described in the [Quickstart Guide](../README.md#quickstart).
3. Open a new terminal and launch the Docker container using the `run_dev.sh` script:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

4. Build and source the workspace:

    ```bash
    cd /workspaces/isaac_ros-dev && \
      colcon build --symlink-install && \
      source install/setup.bash
    ```

5. Please set your camera as shown in the image below, which is on a tripod ~10cm tall and parallel to the ground. Or you can change the static transform in launch file [here](../isaac_ros_bi3d_freespace/launch/isaac_ros_bi3d_freespace_realsense.launch.py#L144-157), according to the placement of your camera with respect to a occupancy grid origin frame.

    <div align="center"><img src="../resources/realsense_camera_position.jpg"  width="400px"/></div>

6. Run the launch file, which launches the example:

    ```bash
    ros2 launch isaac_ros_bi3d_freespace isaac_ros_bi3d_freespace_realsense.launch.py featnet_engine_file_path:=/tmp/models/bi3d/bi3dnet_featnet.plan \
    segnet_engine_file_path:=/tmp/models/bi3d/bi3dnet_segnet.plan \
    max_disparity_values:=16
    ```

7. Open a second terminal and attach to the container:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
    ./scripts/run_dev.sh
    ```

8. Optionally, you can run the visualizer script to visualize the disparity image.

    ```bash
    ros2 run isaac_ros_bi3d isaac_ros_bi3d_visualizer.py --disparity_topic bi3d_mask
    ```

    <div align="center"><img src="../resources/visualizer_realsense.png" width="500px"/></div>

    <div align="center"><img src="../resources/visualizer_realsense_mono_pair.png" width="500px"/></div>
    > Note: For more information on how to interpret the output, refer to the [interpreting the output section](../README.md#interpreting-the-output) of the main readme.

9. Open a third terminal and attach to the container:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
    ./scripts/run_dev.sh
    ```

10. Visualize the occupancy grid in RViz.

    Start RViz:

    ```bash
    rviz2
    ```

    In the left pane, change the **Fixed Frame** to `base_link`.

    In the left pane, click the **Add** button, then select **By topic** followed by **Map** to add the occupancy grid. You should see an ouput similar to the one shown at the top of this page.
