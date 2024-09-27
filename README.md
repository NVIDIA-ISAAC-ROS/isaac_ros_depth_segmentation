# Isaac ROS Depth Segmentation

NVIDIA-accelerated packages for depth segmentation.

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_depth_segmentation/isaac_ros_bi3d_real_opt.gif/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_depth_segmentation/isaac_ros_bi3d_real_opt.gif/" width="500px"/></a></div>

---

## Webinar Available

Learn how to use this package by watching our on-demand webinar: [Using ML Models in ROS 2 to Robustly Estimate Distance to Obstacles](https://gateway.on24.com/wcc/experience/elitenvidiabrill/1407606/3998202/isaac-ros-webinar-series)

---

### Overview

[Isaac ROS Depth Segmentation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_depth_segmentation) provides NVIDIA NVIDIA-accelerated packages for
depth segmentation. The `isaac_ros_bi3d` package uses the
optimized [Bi3D DNN
model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/models/bi3d_proximity_segmentation)
to perform stereo-depth estimation via binary classification, which is
used for depth segmentation. Depth segmentation can be used to
determine whether an obstacle is within a proximity field and to avoid
collisions with obstacles during navigation.

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_depth_segmentation/isaac_ros_bi3d_nodegraph.png/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_depth_segmentation/isaac_ros_bi3d_nodegraph.png/" width="800px"/></a></div>

[Bi3D](https://arxiv.org/abs/2005.07274) is used in a graph of nodes
to provide depth segmentation from a time-synchronized input left
and right stereo image pair. Images to Bi3D need to be rectified and
resized to the appropriate input resolution. The aspect ratio of the
image needs to be maintained; hence, a crop and resize may be required
to maintain the input aspect ratio. The graph for DNN encode, to DNN
inference, to DNN decode is part of the Bi3D node. Inference is
performed using TensorRT, as the Bi3D DNN model is designed to use
optimizations supported by TensorRT.

Compared to other stereo disparity functions, depth segmentation
provides a prediction of whether an obstacle is within a proximity
field, as opposed to continuous depth, while simultaneously predicting
freespace from the ground plane, which other functions typically do not
provide. Also unlike other stereo disparity functions in Isaac ROS,
depth segmentation runs on NVIDIA DLA (deep learning accelerator),
which is separate and independent from the GPU. For more information on
disparity, refer to [this
page](https://en.wikipedia.org/wiki/Binocular_disparity).

> [!Note]
> This DNN is optimized for and evaluated with RGB global shutter camera images,
> and accuracy may vary on monochrome images.

### Isaac ROS NITROS Acceleration

This package is powered by [NVIDIA Isaac Transport for ROS (NITROS)](https://developer.nvidia.com/blog/improve-perception-performance-for-ros-2-applications-with-nvidia-isaac-transport-for-ros/), which leverages type adaptation and negotiation to optimize message formats and dramatically accelerate communication between participating nodes.

### Performance

| Sample Graph<br/><br/>                                                                                                                                                              | Input Size<br/><br/>     | AGX Orin<br/><br/>                                                                                                                                           | Orin NX<br/><br/>                                                                                                                                           | x86_64 w/ RTX 4060 Ti<br/><br/>                                                                                                                                |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Depth Segmentation Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/benchmarks/isaac_ros_bi3d_benchmark/scripts/isaac_ros_bi3d_node.py)<br/><br/><br/><br/> | 576p<br/><br/><br/><br/> | [45.9 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_bi3d_node-agx_orin.json)<br/><br/><br/>76 ms @ 30Hz<br/><br/> | [28.8 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_bi3d_node-orin_nx.json)<br/><br/><br/>92 ms @ 30Hz<br/><br/> | [87.9 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_bi3d_node-nuc_4060ti.json)<br/><br/><br/>35 ms @ 30Hz<br/><br/> |

---

### Documentation

Please visit the [Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_depth_segmentation/index.html) to learn how to use this repository.

---

### Packages

* [`isaac_ros_bi3d`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_depth_segmentation/isaac_ros_bi3d/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_depth_segmentation/isaac_ros_bi3d/index.html#quickstart)
  * [Try More Examples](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_depth_segmentation/isaac_ros_bi3d/index.html#try-more-examples)
  * [Model Preparation](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_depth_segmentation/isaac_ros_bi3d/index.html#model-preparation)
  * [Troubleshooting](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_depth_segmentation/isaac_ros_bi3d/index.html#troubleshooting)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_depth_segmentation/isaac_ros_bi3d/index.html#api)

### Latest

Update 2024-09-26: Update for ZED compatibility
