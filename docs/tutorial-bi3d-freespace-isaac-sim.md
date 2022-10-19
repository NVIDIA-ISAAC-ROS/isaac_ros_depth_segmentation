# Tutorial for Freespace Segmentation with Isaac Sim
<div align="center"><img src="../resources/Isaac_sim_tutorial.gif" width="600px"/></div></br>

## Overview
This tutorial demonstrates how to use a [Isaac Sim](https://developer.nvidia.com/isaac-sim) and [isaac_ros_bi3d_freespace](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_proximity_segmentation) to create a local occupancy grid.

## Tutorial Walkthrough
1. Complete steps 1-7 listed in the [Quickstart section](../README.md#quickstart) of the main README.
2. Install and launch Isaac Sim following the steps in the [Isaac ROS Isaac Sim Setup Guide](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/isaac-sim-sil-setup.md)
3. Open the Isaac ROS Common USD scene (using the **Content** window) located at:
   
   `omniverse://localhost/NVIDIA/Assets/Isaac/2022.1/Isaac/Samples/ROS2/Scenario/carter_warehouse_apriltags_worker.usd`.
   
   And wait for it to load completely.
   > **Note:** To use a different server, replace `localhost` with `<your_nucleus_server>`
4. Go to the **Stage** tab and select `/World/Carter_ROS/ROS_Cameras/ros2_create_camera_right_info`. In the **Property** tab, change the **Compute Node -> Inputs -> stereoOffset -> X** value from `0` to `-175.92`.
    <div align="center"><img src="../resources/Isaac_sim_set_stereo_offset.png" width="500px"/></div></br>
5.  Enable the right camera for a stereo image pair. Go to the **Stage** tab and select `/World/Carter_ROS/ROS_Cameras/enable_camera_right`, then tick the **Condition** checkbox.
    <div align="center"><img src="../resources/Isaac_sim_enable_stereo.png" width="500px"/></div></br>
6. Project the `base_link` frame to the ground floor so that we can anchor our occupancy grid. Go to the **Stage** tab and select `/World/Carter_ROS/chassis_link/base_link`. In the **Property** tab, change the **Transform -> Translate -> Z** value from `0` to `-0.24`.
    <div align="center"><img src="../resources/Isaac_sim_change_base_link.png" width="500px"/></div></br>

7.  Disable the clock reset when simulation is stopped. Go to the **Stage** tab and select `/World/Clock/isaac_read_simulation_time`, then untick the **Reset On Stop** checkbox.
    <div align="center"><img src="../resources/Isaac_sim_disable_clock_reset.png" width="500px"/></div></br>
8.  Press **Play** to start publishing data from the Isaac Sim application.
    <div align="center"><img src="../resources/Isaac_sim_play.png" width="800px"/></div></br>
9.  Open a second terminal and attach to the container:
    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
    ./scripts/run_dev.sh
    ```
10. In the second terminal, start the `isaac_ros_bi3d` node using the launch files:
    ```bash
    ros2 launch isaac_ros_bi3d_freespace isaac_ros_bi3d_freespace_isaac_sim.launch.py \
    featnet_engine_file_path:=/tmp/models/bi3d/bi3dnet_featnet.plan \
    segnet_engine_file_path:=/tmp/models/bi3d/bi3dnet_segnet.plan \
    max_disparity_values:=32
    ```
    You should see a RViz window, as shown below:
    <div align="center"><img src="../resources/Isaac_sim_rviz.png" width="500px"/></div></br>

11. Optionally, you can run the visualizer script to visualize the disparity image.
    ```bash
    ros2 run isaac_ros_bi3d isaac_ros_bi3d_visualizer.py --disparity_topic bi3d_mask
    ```
    <div align="center"><img src="../resources/Visualizer_isaac_sim.png" width="500px"/></div>
