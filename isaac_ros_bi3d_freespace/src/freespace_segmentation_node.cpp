// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_bi3d_freespace/freespace_segmentation_node.hpp"


#include "isaac_ros_nitros_occupancy_grid_type/nitros_occupancy_grid.hpp"
#include "isaac_ros_nitros_disparity_image_type/nitros_disparity_image.hpp"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace bi3d_freespace
{

using nvidia::gxf::optimizer::GraphIOGroupSupportedDataTypesInfoList;

constexpr char INPUT_COMPONENT_KEY[] = "freespace_segmentation/mask_in";
constexpr char INPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_disparity_image_32FC1";
constexpr char INPUT_TOPIC_NAME[] = "bi3d_mask";

constexpr char OUTPUT_COMPONENT_KEY[] = "vault/vault";
constexpr char OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_occupancy_grid";
constexpr char OUTPUT_TOPIC_NAME[] = "freespace_segmentation/occupancy_grid";

constexpr char APP_YAML_FILENAME[] = "config/freespace_segmentation_node.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_bi3d_freespace";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_nitros", "gxf/std/libgxf_std.so"},
  {"isaac_ros_nitros", "gxf/cuda/libgxf_cuda.so"},
  {"isaac_ros_nitros", "gxf/multimedia/libgxf_multimedia.so"},
  {"isaac_ros_nitros", "gxf/libgxf_occupancy_grid_projector.so"},
};
const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {
  "isaac_ros_bi3d_freespace"
};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {};
const std::vector<std::string> GENERATOR_RULE_FILENAMES = {};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
const nitros::NitrosPublisherSubscriberConfigMap CONFIG_MAP = {
  {INPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = INPUT_TOPIC_NAME,
    }
  },
  {OUTPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = OUTPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = OUTPUT_TOPIC_NAME,
      .frame_id_source_key = INPUT_COMPONENT_KEY,
    }
  },
};
#pragma GCC diagnostic pop

FreespaceSegmentationNode::FreespaceSegmentationNode(const rclcpp::NodeOptions & options)
: nitros::NitrosNode(
    options,
    APP_YAML_FILENAME,
    CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES,
    EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES,
    EXTENSIONS,
    PACKAGE_NAME),
  tf_buffer_{std::make_unique<tf2_ros::Buffer>(this->get_clock())},
  transform_listener_{std::make_shared<tf2_ros::TransformListener>(*tf_buffer_)},
  base_link_frame_(declare_parameter<std::string>("base_link_frame", "base_link")),
  camera_frame_(declare_parameter<std::string>("camera_frame", "camera")),
  f_x_(declare_parameter<double>("f_x", 0.0)),
  f_y_(declare_parameter<double>("f_y", 0.0)),
  grid_height_(declare_parameter<int>("grid_height", 100)),
  grid_width_(declare_parameter<int>("grid_width", 100)),
  grid_resolution_(declare_parameter<double>("grid_resolution", 0.01))
{
  RCLCPP_DEBUG(get_logger(), "[FreespaceSegmentationNode] Initializing FreespaceSegmentationNode");

  config_map_[OUTPUT_COMPONENT_KEY].callback =
    [this]([[maybe_unused]] const gxf_context_t context, nitros::NitrosTypeBase & msg) {
      msg.frame_id = base_link_frame_;
    };

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosDisparityImage>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosOccupancyGrid>();

  RCLCPP_DEBUG(get_logger(), "[FreespaceSegmentationNode] Constructor");
  if (f_x_ <= 0 || f_y_ <= 0) {
    RCLCPP_ERROR(
      get_logger(), "[FreespaceSegmentationNode] Set the focal length before running.");
    throw std::invalid_argument(
            "[FreespaceSegmentationNode] Invalid focal length "
            "fx and fy should be valid focal lengths in pixels.");
  }

  startNitrosNode();
}

void FreespaceSegmentationNode::postLoadGraphCallback()
{
  RCLCPP_DEBUG(get_logger(), "[FreespaceSegmentationNode] postLoadGraphCallback().");

  // Collect tf2 transform at initialization time
  RCLCPP_INFO(this->get_logger(), "Waiting for tf2 transform...");

  rclcpp::Rate rate{1};

  bool transform_exists = false;
  while (!transform_exists) {
    if (!rclcpp::ok()) {
      RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for the service. Exiting.");
      return;
    }
    try {
      geometry_msgs::msg::TransformStamped t = tf_buffer_->lookupTransform(
        base_link_frame_, camera_frame_, this->now());

      projection_transform_param_ = {
        t.transform.translation.x,
        t.transform.translation.y,
        t.transform.translation.z,

        t.transform.rotation.x,
        t.transform.rotation.y,
        t.transform.rotation.z,
        t.transform.rotation.w,
      };

      transform_exists = true;
    } catch (const tf2::TransformException & ex) {
      RCLCPP_ERROR(
        this->get_logger(), "Could not transform %s to %s: %s",
        base_link_frame_.c_str(), camera_frame_.c_str(), ex.what());
    }
    rate.sleep();
  }
  RCLCPP_INFO(this->get_logger(), "Transform secured");

  getNitrosContext().setParameter1DFloat64Vector(
    "freespace_segmentation", "nvidia::isaac_ros::freespace_segmentation::OccupancyGridProjector",
    "projection_transform",
    projection_transform_param_);

  intrinsics_param_ = {f_x_, f_y_};
  getNitrosContext().setParameter1DFloat64Vector(
    "freespace_segmentation", "nvidia::isaac_ros::freespace_segmentation::OccupancyGridProjector",
    "intrinsics",
    intrinsics_param_);

  getNitrosContext().setParameterInt32(
    "freespace_segmentation", "nvidia::isaac_ros::freespace_segmentation::OccupancyGridProjector",
    "grid_height", grid_height_);
  getNitrosContext().setParameterInt32(
    "freespace_segmentation", "nvidia::isaac_ros::freespace_segmentation::OccupancyGridProjector",
    "grid_width", grid_width_);
  getNitrosContext().setParameterFloat64(
    "freespace_segmentation", "nvidia::isaac_ros::freespace_segmentation::OccupancyGridProjector",
    "grid_resolution", grid_resolution_);
}

FreespaceSegmentationNode::~FreespaceSegmentationNode() {}

}  // namespace bi3d_freespace
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::bi3d_freespace::FreespaceSegmentationNode)
