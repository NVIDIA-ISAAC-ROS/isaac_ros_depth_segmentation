# Isaac ROS Proximity Segmentation

<div align="center"><img alt="Isaac ROS Proximity Segmentation Sample Output" src="resources/isaac_ros_bi3d_real_opt.gif" width="500px"/></div>

---
## Webinar Available
Learn how to use this package by watching our on-demand webinar: [Using ML Models in ROS2 to Robustly Estimate Distance to Obstacles](https://gateway.on24.com/wcc/experience/elitenvidiabrill/1407606/3998202/isaac-ros-webinar-series)

---

## Overview

This repository provides NVIDIA hardware-accelerated packages for proximity segmentation. The `isaac_ros_bi3d` package uses an optimized [Bi3D](https://arxiv.org/abs/2005.07274) model to perform stereo-depth estimation via binary classification, which is used for proximity segmentation. Proximity segmentation can be used to determine whether an obstacle is within a proximity field and to avoid collisions with obstacles during navigation.

Proximity segmentation predicts freespace from the ground plane, eliminating the need to perform ground plane removal from the segmentation image. The `isaac_ros_bi3d_freespace` package leverages this functionality to produce an occupancy grid that indicates freespace in the neighborhood of the robot. This camera-based solution offers a number of appealing advantages over traditional 360 lidar occupancy scanning, including better detection of low-profile obstacles.

Compared to other stereo disparity functions, proximity segmentation is unique in that it provides a prediction of whether an obstacle is within a proximity field, instead of continuous depth, while simultaneously predicting freespace from the ground plane, which other functions typically do not provide. Proximity segmentation is diverse relative to other stereo disparity functions in Isaac ROS, as it is a different algorithm that runs on NVIDIA DLA (deep learning accelerator), which is separate and independent from the GPU. For more information on disparity, refer to [this page](https://en.wikipedia.org/wiki/Binocular_disparity).

### Isaac ROS NITROS Acceleration

This package is powered by [NVIDIA Isaac Transport for ROS (NITROS)](https://developer.nvidia.com/blog/improve-perception-performance-for-ros-2-applications-with-nvidia-isaac-transport-for-ros/), which leverages type adaptation and negotiation to optimize message formats and dramatically accelerate communication between participating nodes.

## Performance

The following are the benchmark performance results of the prepared pipelines in this package, by supported platform:

| Pipeline                      | AGX Orin            | Orin Nano        | x86_64 w/ RTX 3060 Ti |
| ----------------------------- | ------------------- | ---------------- | --------------------- |
| Bi3D Stereo Node (576p) (DLA) | 62 fps <br> 46ms    | N/A              | N/A                   |
| Bi3D Stereo Node (576p) (GPU) | 81 fps <br> 61ms    | 25 fps <br> 65ms | 145 fps <br> 386ms    |
| Freespace Segmentation        | 1145 fps <br> 1.3ms | 725 fps <br> 2ms | 1490 fps <br> 0.3ms   |

These data have been collected per the methodology described [here](https://github.com/NVIDIA-ISAAC-ROS/.github/blob/main/profile/performance-summary.md#methodology).

## Table of Contents

- [Isaac ROS Proximity Segmentation](#isaac-ros-proximity-segmentation)
  - [Webinar Available](#webinar-available)
  - [Overview](#overview)
    - [Isaac ROS NITROS Acceleration](#isaac-ros-nitros-acceleration)
  - [Performance](#performance)
  - [Table of Contents](#table-of-contents)
  - [Latest Update](#latest-update)
  - [Supported Platforms](#supported-platforms)
    - [Docker](#docker)
  - [Quickstart](#quickstart)
  - [Next Steps](#next-steps)
    - [Try More Examples](#try-more-examples)
    - [Use Different Models](#use-different-models)
    - [Customize your Dev Environment](#customize-your-dev-environment)
  - [Model Preparation](#model-preparation)
    - [Download Pre-trained Models (.onnx) from NGC](#download-pre-trained-models-onnx-from-ngc)
    - [Convert the Pre-trained Models (.onnx) to TensorRT Engine Plans](#convert-the-pre-trained-models-onnx-to-tensorrt-engine-plans)
      - [Generating Engine Plans for Jetson](#generating-engine-plans-for-jetson)
      - [Generating Engine Plans for x86_64](#generating-engine-plans-for-x86_64)
  - [Package Reference](#package-reference)
    - [`isaac_ros_bi3d`](#isaac_ros_bi3d)
      - [Bi3D Overview](#bi3d-overview)
      - [Usage](#usage)
      - [Interpreting the Output](#interpreting-the-output)
      - [ROS Parameters](#ros-parameters)
      - [ROS Topics Subscribed](#ros-topics-subscribed)
      - [ROS Topics Published](#ros-topics-published)
    - [`isaac_ros_bi3d_freespace`](#isaac_ros_bi3d_freespace)
      - [Usage](#usage-1)
      - [ROS Parameters](#ros-parameters-1)
      - [ROS Topics Subscribed](#ros-topics-subscribed-1)
      - [ROS Topics Published](#ros-topics-published-1)
  - [Troubleshooting](#troubleshooting)
    - [Isaac ROS Troubleshooting](#isaac-ros-troubleshooting)
    - [DNN and Triton Troubleshooting](#dnn-and-triton-troubleshooting)
  - [Updates](#updates)

## Latest Update

Update 2022-10-19: Add Freespace Segmentation functionality to produce occupancy grids from Bi3D's output

## Supported Platforms

This package is designed and tested to be compatible with ROS2 Humble running on [Jetson](https://developer.nvidia.com/embedded-computing) or an x86_64 system with an NVIDIA GPU.

> **Note**: Versions of ROS2 earlier than Humble are **not** supported. This package depends on specific ROS2 implementation features that were only introduced beginning with the Humble release.

| Platform | Hardware                                                                                                                                                                                                 | Software                                                                                                             | Notes                                                                                                                                                                                       |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Jetson   | [Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) <br> [Jetson Xavier](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-agx-xavier/) | [JetPack 5.0.2](https://developer.nvidia.com/embedded/jetpack)                                                       | For best performance, ensure that the [power settings](https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance.html) are configured appropriately. |
| x86_64   | NVIDIA GPU                                                                                                                                                                                               | [Ubuntu 20.04+](https://releases.ubuntu.com/20.04/) <br> [CUDA 11.6.1+](https://developer.nvidia.com/cuda-downloads) |

### Docker

To simplify development, we strongly recommend leveraging the Isaac ROS Dev Docker images by following [these steps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/dev-env-setup.md). This will streamline your development environment setup with the correct versions of dependencies on both Jetson and x86_64 platforms.

> **Note:** All Isaac ROS quick start guides, tutorials, and examples have been designed with the Isaac ROS Docker images as a prerequisite.

## Quickstart

1. Set up your development environment by following the instructions [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/dev-env-setup.md).

2. Clone this repository and its dependencies under `~/workspaces/isaac_ros-dev/src`.

      ```bash
      cd ~/workspaces/isaac_ros-dev/src && 
      git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common && 
      git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros && 
      git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline &&
      git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_proximity_segmentation
      ```

3. Pull down a rosbag of sample data:

      ```bash
      cd ~/workspaces/isaac_ros-dev/src/isaac_ros_proximity_segmentation && 
      git lfs pull -X "" -I "resources/rosbags/bi3dnode_rosbag"
      ```

4. Launch the Docker container using the `run_dev.sh` script:

      ```bash
      cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common &&
      ./scripts/run_dev.sh
      ```

5. Download model files for Bi3D (refer to the [Model Preparation](#model-preparation) section for more information):

      ```bash
      mkdir -p /tmp/models/bi3d &&
      cd /tmp/models/bi3d &&
      wget 'https://api.ngc.nvidia.com/v2/models/nvidia/isaac/bi3d_proximity_segmentation/versions/2.0.0/files/featnet.onnx' &&
      wget 'https://api.ngc.nvidia.com/v2/models/nvidia/isaac/bi3d_proximity_segmentation/versions/2.0.0/files/segnet.onnx'
      ```

6. Convert the `.onnx` model files to TensorRT engine plan files (refer to the [Model Preparation](#model-preparation) section for more information):

    If using Jetson (Generate engine plans with DLA support enabled):

      ```bash
      /usr/src/tensorrt/bin/trtexec --saveEngine=/tmp/models/bi3d/bi3dnet_featnet.plan \
      --onnx=/tmp/models/bi3d/featnet.onnx \
      --int8 --useDLACore=0 &&
      /usr/src/tensorrt/bin/trtexec --saveEngine=/tmp/models/bi3d/bi3dnet_segnet.plan \
      --onnx=/tmp/models/bi3d/segnet.onnx \
      --int8 --useDLACore=0
      ```

    If using x86_64:

      ```bash
      /usr/src/tensorrt/bin/trtexec --saveEngine=/tmp/models/bi3d/bi3dnet_featnet.plan \
      --onnx=/tmp/models/bi3d/featnet.onnx --int8 &&
      /usr/src/tensorrt/bin/trtexec --saveEngine=/tmp/models/bi3d/bi3dnet_segnet.plan \
      --onnx=/tmp/models/bi3d/segnet.onnx --int8
      ```

    > **Note:** The engine plans generated using the x86_64 commands will also work on Jetson, but performance will be reduced.

7. Build and source the workspace:  

      ```bash
      cd /workspaces/isaac_ros-dev &&
      colcon build --symlink-install &&
      source install/setup.bash
      ```

8. (Optional) Run tests to verify complete and correct installation:  

      ```bash
      colcon test --executor sequential
      ```

9. Run the launch file to spin up a demo of this package:

      ```bash
      ros2 launch isaac_ros_bi3d_freespace isaac_ros_bi3d_freespace.launch.py featnet_engine_file_path:=/tmp/models/bi3d/bi3dnet_featnet.plan \
      segnet_engine_file_path:=/tmp/models/bi3d/bi3dnet_segnet.plan \
      max_disparity_values:=10
      ```

10. Open a **second** terminal inside the Docker container:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

11. Play the rosbag file to simulate image streams from the cameras:

    ```bash
    ros2 bag play --loop src/isaac_ros_proximity_segmentation/resources/rosbags/bi3dnode_rosbag
    ```

12. Open a **third** terminal inside the Docker container:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

13. Visualize the occupancy grid in RViz.

    Start RViz:

    ```bash
    rviz2
    ```

    In the left pane, change **Fixed Frame** to `base_link`.

    In the left pane, click the **Add** button, then select **By topic** followed by **Map** to add the occupancy grid.

    <div align="center"><img alt="RViz Output" src="resources/Rviz_quickstart.png" width="500px"/></div>

## Next Steps

### Try More Examples

To continue your exploration, check out the following suggested examples:

| Example                                                                                                  | Dependencies                                                                                                         |
| -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| [Zone detection for an autonomous mobile robot (AMR)](./docs/bi3d-example.md)                            | --                                                                                                                   |
| [Tutorial with RealSense, Bi3D, and Freespace Segmentation](./docs/tutorial-bi3d-freespace-realsense.md) | [`realsense-ros`](https://github.com/NVIDIA-ISAAC-ROS/.github/blob/main/profile/realsense-setup.md)                  |
| [Tutorial for Bi3D with Isaac Sim](./docs/tutorial-bi3d-isaac-sim.md)                                    | --                                                                                                                   |
| [Tutorial for Freespace Segmentation with Isaac Sim](./docs/tutorial-bi3d-freespace-isaac-sim.md)        | --                                                                                                                   |

### Use Different Models

Click [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/blob/main/docs/model-preparation.md) for more information about how to use NGC models.

### Customize your Dev Environment

To customize your development environment, reference [this guide](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/modify-dockerfile.md).

## Model Preparation

### Download Pre-trained Models (.onnx) from NGC

The following steps show how to download pretrained Bi3D DNN inference models.

1. The following model files must be downloaded to perform Bi3D inference. From **File Browser** on the **Bi3D** [page](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/models/bi3d_proximity_segmentation), select the following `.onnx` model files in the **FILE** list and copy the `wget` command by clicking **...** in the **ACTIONS** column:
   - `featnet.onnx`
   - `segnet.onnx`

2. Run each of the copied commands in a terminal to download the ONNX model file, as shown in the example below:

   ```bash
   wget 'https://api.ngc.nvidia.com/v2/models/nvidia/isaac/bi3d_proximity_segmentation/versions/2.0.0/files/featnet.onnx' &&
   wget 'https://api.ngc.nvidia.com/v2/models/nvidia/isaac/bi3d_proximity_segmentation/versions/2.0.0/files/segnet.onnx'
   ```

- Bi3D Featnet is a network that extracts features from stereo images.

- Bi3D Segnet is an encoder-decoder segmentation network that generates a binary segmentation confidence map.

### Convert the Pre-trained Models (.onnx) to TensorRT Engine Plans

`trtexec` is used to convert pre-trained models (`.onnx`) to the TensorRT engine plan and is included in the Isaac ROS docker container under `/usr/src/tensorrt/bin/trtexec`.

> **Tip:** Use `/usr/src/tensorrt/bin/trtexec -h` for more information on using the tool.

#### Generating Engine Plans for Jetson  

  ```bash
  /usr/src/tensorrt/bin/trtexec --onnx=<PATH_TO_ONNX_MODEL_FILE> --saveEngine=<PATH_TO_WHERE_TO_SAVE_ENGINE_PLAN> --useDLACore=<SET_CORE_TO_ENABLE_DLA> --int8
  ```

#### Generating Engine Plans for x86_64  

  ```bash
  /usr/src/tensorrt/bin/trtexec --onnx=<PATH_TO_ONNX_MODEL_FILE> --saveEngine=<PATH_TO_WHERE_TO_SAVE_ENGINE_PLAN> --int8
  ```

## Package Reference

### `isaac_ros_bi3d`

#### Bi3D Overview

Bi3D predicts if an obstacle is within a given proximity field via a series of binary classifications; the binary classification per pixel determines if the pixel is in front of or behind the proximity field. As such, Bi3D is differentiated from other stereo disparity functions which output continuous [disparity](https://en.wikipedia.org/wiki/Binocular_disparity). Bi3D allows you to increase the diversity of functions used for obstacle detection and improve hardware diversity because Isaac ROS Proximity Segmentation is optimized to run on NVIDIA DLA hardware, which is separate from the GPU. In the form presented here, Bi3D is intended to provide proximity detections, and is not a replacement for the continuous depth estimation provided by Isaac ROS DNN Stereo Disparity.

> **Note:** This DNN is optimized for and evaluated with RGB global shutter camera images, and accuracy may vary on monochrome images.

#### Usage

  ```bash
  ros2 launch isaac_ros_bi3d isaac_ros_bi3d.launch.py featnet_engine_file_path:=<PATH_TO_FEATNET_ENGINE> \
  segnet_engine_file_path:=<PATH_TO_SEGNET_ENGINE \
  max_disparity_values:=<MAX_NUMBER_OF_DISPARITY_VALUES_USED>
  ```

#### Interpreting the Output

The `isaas_ros_bi3d` package outputs a disparity image given a list of disparity values (planes). Each pixel of the output image that is not freespace is set to the value of the closest disparity plane (largest disparity value) that the pixel is deemed to be in front of. Each pixel that is predicted to be freespace is set to 0 (the furthest disparity/smallest disparity value). Freespace is defined as the region from the bottom of the image, up to the first pixel above which is not the ground plane. To find the boundary between freespace and not-freespace, one may start from the bottom of the image and, per column, find the first pixel that is not the ground plane. In the below example, the freespace of the image is shown in black:

<div align="center"><img alt="Freespace Example (Original)" src="resources/freespace_example_real.png" width="400px"/></div>
<div align="center"><img alt="Freespace Example (Segmented)" src="resources/freespace_example_segmented.png" width="400px"/></div>

The prediction of freespace eliminates the need for ground plane removal in the output image as a post-processing step, which is often applied to other stereo disparity functions. The output of `isaas_ros_bi3d` can be used to check if any pixels within the image breach a given proximity field by checking the values of all pixels. If a pixel value (disparity value) is larger than the disparity plane defining the proximity field, then it has breached that field. If a pixel does not breach any of the provided disparty planes, it is assigned a value of 0.

#### ROS Parameters

| ROS Parameter              | Type                   | Default                    | Description                                                                                                                                                                                                                                                                                                               |
| -------------------------- | ---------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `featnet_engine_file_path` | `std::string`          | `'path_to_featnet_engine'` | The path to the Bi3D Featnet engine plan                                                                                                                                                                                                                                                                                  |
| `segnet_engine_file_path`  | `std::string`          | `'path_to_segnet_engine'`  | The path to the Bi3D Segnet engine plan                                                                                                                                                                                                                                                                                   |
| `max_disparity_values`     | `int64_t`              | `64`                       | The maximum number of disparity values used for Bi3D inference. Isaac ROS Proximity Segmentation supports up to a theoretical maximum of 64 disparity values during inference. However, the maximum length of disparities that a user may run in practice is dependent on the user's hardware and availability of memory. |
| `disparity_values`         | `std::vector<int64_t>` | `{10, 20, 30, 40, 50, 60}` | The specific threshold disparity values used for Bi3D inference. The number of disparity values must not exceed the value set in the `max_disparity_values` ROS parameter.                                                                                                                                                |

#### ROS Topics Subscribed

| ROS Topic          | Interface                                                                                            | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ------------------ | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `left_image_bi3d`  | [sensor_msgs/Image](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/Image.msg) | 1. The message must be a ROS `sensor_msgs/Image` of size **W=960, H=576** with `rgb8` image encoding. <br><br> 2. There should only be a single publisher node publishing to `left_image_bi3d`. Timing behaviour with multiple publishers is not guaranteed by Bi3DNode and inference may not be performed on correct image pairs. Bi3D will process input pairs on a first available basis. Use a separate instance of Bi3DNode for each unique scene (publisher) that you wish to process.   |
| `right_image_bi3d` | [sensor_msgs/Image](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/Image.msg) | 1. The message must be a ROS `sensor_msgs/Image` of size **W=960, H=576** with `rgb8` image encoding. <br><br> 2. There should only be a single publisher node publishing to `right_image_bi3d`. Timing behaviour with multiple publishers is not guaranteed by Bi3DNode and inference may not be performed on correct image pairs. Bi3D will process inputs pairs on a first available basis. Use a separate instance of Bi3DNode for each unique scene (publisher) that you wish to process. |  |
> Note: The images on input topics (`left_image_bi3d` and `right_image_bi3d`) should be a color image in `rgb8` format.

#### ROS Topics Published

| ROS Topic                         | Interface                                                                                                                                                                                       | Description                                                                                                                                                                                                                                                                                                       |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bi3d_node/bi3d_output`           | [stereo_msgs/DisparityImage](https://github.com/ros2/common_interfaces/blob/humble/stereo_msgs/msg/DisparityImage.msg)                                                                          | The proximity segmentation of Bi3D given as a disparity image. For pixels not deemed freespace, their value is set to the closest (largest) disparity plane that is breached. A pixel value is set to 0 if it doesn't breach any disparity plane or if it is freespace. <br><br> Output Resolution: 960x576 (WxH) |
| `bi3d_node/bi3d_disparity_values` | [isaac_ros_bi3d_interfaces/Bi3DInferenceParametersArray](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/isaac_ros_bi3d_interfaces/msg/Bi3DInferenceParametersArray.msg)            | The disparity values used for Bi3D inference. The timestamp is matched to the timestamp in the correpsonding output image from  `bi3d_node/bi3d_output`                                                                                                                                                           |

### `isaac_ros_bi3d_freespace`

#### Usage

```bash
ros2 launch isaac_ros_bi3d_freespace isaac_ros_freespace_segmentation.launch.py base_link_frame:=<"name of base link"> camera_frame:=<"name of camera frame"> s_x:=<"pixel pitch in x dimension"> s_y:=<"pixel pitch in y dimension"> f:=<"focal length"> grid_width:=<"desired grid width"> grid_height:=<"desired grid height"> grid_resolution:=<"desired grid resolution">
```

#### ROS Parameters

| ROS Parameter     | Type          | Default     | Description                                                                     |
| ----------------- | ------------- | ----------- | ------------------------------------------------------------------------------- |
| `base_link_frame` | `std::string` | `base_link` | The name of the `tf2` frame attached to the robot base                          |
| `camera_frame`    | `std::string` | `camera`    | The name of the `tf2` frame attached to the camera                              |
| `s_x`             | `double`      | `1e5`       | The pitch of pixels on the image sensor in the X dimension, in pixels per meter |
| `s_y`             | `double`      | `1e5`       | The pitch of pixels on the image sensor in the Y dimension, in pixels per meter |
| `f`               | `double`      | `1e-2`      | The focal length of the camera, in meters                                       |
| `grid_width`      | `int`         | `100`       | The width of the output occupancy grid, in number of cells                      |
| `grid_height`     | `int`         | `100`       | The height of the output occupancy grid, in number of cells                     |
| `grid_resolution` | `double`      | `0.01`      | The resolution of the output occupancy grid, in meters per cell                 |

#### ROS Topics Subscribed

| ROS Topic                        | Interface                                                                                                              | Description                                                                  |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `freespace_segmentation/mask_in` | [stereo_msgs/DisparityImage](https://github.com/ros2/common_interfaces/blob/humble/stereo_msgs/msg/DisparityImage.msg) | The input disparity image, with pixels corresponding to ground labelled as 0 |

> **Limitation:** For all input images, both the height and width must be an even number of pixels.

#### ROS Topics Published

| ROS Topic                               | Interface                                                                                                      | Description                                               |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `freespace_segmentation/occupancy_grid` | [nav_msgs/OccupancyGrid](https://github.com/ros2/common_interfaces/blob/humble/nav_msgs/msg/OccupancyGrid.msg) | The output occupancy grid, with cells marked as 0 if free |

## Troubleshooting

### Isaac ROS Troubleshooting

Check [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/troubleshooting.md) for solutions to problems with Isaac ROS.

### DNN and Triton Troubleshooting

Check [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/blob/main/docs/troubleshooting.md) for solutions to problems with using DNN models and Triton.

## Updates

| Date       | Changes                                                                                |
| ---------- | -------------------------------------------------------------------------------------- |
| 2022-10-19 | Add Freespace Segmentation functionality to produce occupancy grids from Bi3D's output |
| 2022-08-31 | Update to use latest model and to be compatible with JetPack 5.0.2                     |
| 2022-06-30 | Initial release                                                                        |
