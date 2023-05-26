// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_bi3d/bi3d_node.hpp"

#include <cstdio>
#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <algorithm>

#include "isaac_ros_bi3d_interfaces/msg/bi3_d_inference_parameters_array.hpp"
#include "isaac_ros_nitros_bi3d_inference_param_array_type/nitros_bi3d_inference_param_array.hpp"
#include "isaac_ros_nitros_disparity_image_type/nitros_disparity_image.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace bi3d
{

using nvidia::gxf::optimizer::GraphIOGroupSupportedDataTypesInfoList;

constexpr char INPUT_IMAGE_DEFAULT_TENSOR_FORMAT[] = "nitros_image_rgb8";

constexpr char INPUT_LEFT_IMAGE_COMPONENT_KEY[] = "sync/data_receiver_left";
constexpr char INPUT_LEFT_IMAGE_TOPIC_NAME[] = "left_image_bi3d";
constexpr char INPUT_RIGHT_IMAGE_COMPONENT_KEY[] = "sync/data_receiver_right";
constexpr char INPUT_RIGHT_IMAGE_TOPIC_NAME[] = "right_image_bi3d";

constexpr char INPUT_DISPARITY_COMPONENT_KEY[] = "disparity_roundrobin/data_receiver";

constexpr char OUTPUT_BI3D_KEY[] = "bi3d_output_sink/sink";
constexpr char OUTPUT_BI3D_DEFAULT_TENSOR_FORMAT[] = "nitros_disparity_image_32FC1";
constexpr char OUTPUT_BI3D_TOPIC_NAME[] = "bi3d_node/bi3d_output";

constexpr char APP_YAML_FILENAME[] = "config/bi3d_node.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_bi3d";

const uint64_t BI3D_BLOCK_SIZE = 2211840;

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_gxf", "gxf/lib/std/libgxf_std.so"},
  {"isaac_ros_gxf", "gxf/lib/cuda/libgxf_cuda.so"},
  {"isaac_ros_gxf", "gxf/lib/multimedia/libgxf_multimedia.so"},
  {"isaac_ros_gxf", "gxf/lib/libgxf_synchronization.so"},
  {"isaac_ros_bi3d", "gxf/lib/bi3d/libgxf_cvcore_bi3d.so"},
  {"isaac_ros_bi3d", "gxf/lib/bi3d/libgxf_bi3d_postprocessor.so"}
};
const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {
  "isaac_ros_bi3d",
};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {};
const std::vector<std::string> GENERATOR_RULE_FILENAMES = {};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
const nitros::NitrosPublisherSubscriberConfigMap CONFIG_MAP = {
  {INPUT_LEFT_IMAGE_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_IMAGE_DEFAULT_TENSOR_FORMAT,
      .topic_name = INPUT_LEFT_IMAGE_TOPIC_NAME,
    }
  },
  {INPUT_RIGHT_IMAGE_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_IMAGE_DEFAULT_TENSOR_FORMAT,
      .topic_name = INPUT_RIGHT_IMAGE_TOPIC_NAME,
    }
  },
  {INPUT_DISPARITY_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NOOP
    }
  },
  {OUTPUT_BI3D_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = OUTPUT_BI3D_DEFAULT_TENSOR_FORMAT,
      .topic_name = OUTPUT_BI3D_TOPIC_NAME,
      .frame_id_source_key = INPUT_LEFT_IMAGE_COMPONENT_KEY,
    }
  }
};
#pragma GCC diagnostic pop

void Bi3DNode::Bi3DVideoBufferNameCallback(
  const gxf_context_t context,
  nitros::NitrosTypeBase & msg, std::string name, bool pub_disp_values)
{
  gxf_tid_t video_buffer_tid;
  auto result_tid = GxfComponentTypeId(
    context, "nvidia::gxf::VideoBuffer", &video_buffer_tid);
  if (result_tid != GXF_SUCCESS) {
    std::string error_msg = "[Bi3DNode] Bi3DVideoBufferNameCallback Error: Could not get type ID";
    RCLCPP_ERROR(get_logger(), error_msg.c_str());
    throw std::runtime_error(error_msg.c_str());
  }

  gxf_uid_t video_buffer_cid;
  auto result_cid = GxfComponentFind(
    context,
    msg.handle, video_buffer_tid, nullptr, nullptr, &video_buffer_cid);
  if (result_cid != GXF_SUCCESS) {
    std::string error_msg =
      "[Bi3DNode] Bi3DVideoBufferNameCallback Error: Could not get component ID";
    RCLCPP_ERROR(get_logger(), error_msg.c_str());
    throw std::runtime_error(error_msg.c_str());
  }

  // Set VideoBuffer name
  GxfParameterSetStr(
    context, video_buffer_cid, kInternalNameParameterKey, name.c_str());

  if (pub_disp_values) {
    using bi3d_ros_type = isaac_ros_bi3d_interfaces::msg::Bi3DInferenceParametersArray;
    using bi3d_nitros_type = nitros::NitrosBi3DInferenceParamArray;

    auto disp_values_ros_msg = std::make_unique<bi3d_ros_type>();
    get_parameter("disparity_values", disparity_values_);
    disp_values_ros_msg->disparity_values = std::vector<int32_t>(
      disparity_values_.begin(), disparity_values_.end());

    bi3d_nitros_type disp_values_nitros_msg = bi3d_nitros_type();
    rclcpp::TypeAdapter<bi3d_nitros_type, bi3d_ros_type>::convert_to_custom(
      *disp_values_ros_msg, disp_values_nitros_msg);

    auto disparity_sub = findNitrosSubscriber(
      {"nvidia::gxf::DoubleBufferReceiver", "data_receiver", "disparity_roundrobin"});

    if (disparity_sub == nullptr) {
      std::string error_msg =
        "[Bi3DNode] Could not find receiver of the disparity values";
      RCLCPP_ERROR(get_logger(), error_msg.c_str());
      throw std::runtime_error(error_msg.c_str());
    }

    disparity_sub->pushEntity(disp_values_nitros_msg.handle);
  }
}

Bi3DNode::Bi3DNode(const rclcpp::NodeOptions & options)
: nitros::NitrosNode(
    options,
    APP_YAML_FILENAME,
    CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES,
    EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES,
    EXTENSIONS,
    PACKAGE_NAME),

  // Bi3D model input parameters
  featnet_engine_file_path_(declare_parameter<std::string>(
      "featnet_engine_file_path",
      "path_to_featnet_engine")),
  featnet_input_layers_name_(declare_parameter<std::vector<std::string>>(
      "featnet_input_layers_name",
      {"input.1"})),
  featnet_output_layers_name_(declare_parameter<std::vector<std::string>>(
      "featnet_output_layers_name", {"97"})),

  segnet_engine_file_path_(declare_parameter<std::string>(
      "segnet_engine_file_path",
      "path_to_segnet_engine")),
  segnet_input_layers_name_(declare_parameter<std::vector<std::string>>(
      "segnet_input_layers_name",
      {"input.1"})),
  segnet_output_layers_name_(declare_parameter<std::vector<std::string>>(
      "segnet_output_layers_name",
      {"278"})),

  // Bi3D extra parameters
  max_disparity_values_(declare_parameter<int64_t>("max_disparity_values", 64)),
  disparity_values_(declare_parameter<std::vector<int64_t>>(
      "disparity_values", {10, 20, 30, 40, 50, 60}))
{
  RCLCPP_DEBUG(get_logger(), "[Bi3DNode] Initializing Bi3DNode");

  if (featnet_engine_file_path_.empty()) {
    throw std::invalid_argument("[Bi3DNode] Empty featnet_engine_file_path");
  } else if (segnet_engine_file_path_.empty()) {
    throw std::invalid_argument("[Bi3DNode] Empty segnet_engine_file_path");
  }

  if (disparity_values_.size() > static_cast<size_t>(max_disparity_values_)) {
    throw std::invalid_argument(
            "[Bi3DNode] Invalid disparity values: "
            "the number of disparity values (" +
            std::to_string(disparity_values_.size()) + ") cannot exceed the maximum "
            "(max_disparity_values = " + std::to_string(max_disparity_values_) + ")");
  }

  // Add callback functions for setting input VideoBuffer to correct name
  config_map_[INPUT_LEFT_IMAGE_COMPONENT_KEY].callback =
    std::bind(
    &Bi3DNode::Bi3DVideoBufferNameCallback, this,
    std::placeholders::_1, std::placeholders::_2, "left_image", true);
  config_map_[INPUT_RIGHT_IMAGE_COMPONENT_KEY].callback =
    std::bind(
    &Bi3DNode::Bi3DVideoBufferNameCallback, this,
    std::placeholders::_1, std::placeholders::_2, "right_image", false);

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosBi3DInferenceParamArray>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosDisparityImage>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosImage>();

  // This callback will get triggered when there is an attempt to change the parameter
  auto param_change_callback =
    [this](std::vector<rclcpp::Parameter> parameters, int64_t max_disparity_values)
    {
      auto result = rcl_interfaces::msg::SetParametersResult();
      result.successful = true;
      for (auto parameter : parameters) {
        const rclcpp::ParameterType parameter_type = parameter.get_type();
        const std::string param_name = parameter.get_name();

        if (rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY == parameter_type &&
          param_name == "disparity_values")
        {
          if (parameter.as_integer_array().size() > static_cast<size_t>(max_disparity_values)) {
            RCLCPP_ERROR(
              this->get_logger(),
              "[Bi3DNode] Failed to set parameter %s at runtime: "
              "the number of disparity values (%s) exceeds the configured maximum (%ld)",
              parameter.get_name().c_str(),
              parameter.value_to_string().c_str(),
              max_disparity_values
            );
            result.reason = "The number of disparity values exceeds the configured maximum";
            result.successful = false;
          } else {
            RCLCPP_INFO(
              this->get_logger(),
              "[Bi3DNode] Parameter '%s' has changed and is now: %s",
              parameter.get_name().c_str(),
              parameter.value_to_string().c_str()
            );
            result.successful &= true;
          }
        } else {
          RCLCPP_WARN(
            this->get_logger(),
            "[Bi3DNode] Changing value for %s parameter is not allowed during runtime",
            parameter.get_name().c_str()
          );
          result.reason = "Changing value for this parameter is not allowed";
          result.successful = false;
        }
      }
      return result;
    };
  on_set_param_cb_handle_ = this->add_on_set_parameters_callback(
    std::bind(param_change_callback, std::placeholders::_1, max_disparity_values_));

  startNitrosNode();
}

void Bi3DNode::preLoadGraphCallback() {}

void Bi3DNode::postLoadGraphCallback()
{
  // Foward Bi3DNode parameters to GXF component
  // Bi3D model input parameters
  getNitrosContext().setParameterStr(
    "bi3d_dla0", "nvidia::cvcore::Bi3D", "featnet_engine_file_path", featnet_engine_file_path_);
  getNitrosContext().setParameter1DStrVector(
    "bi3d_dla0", "nvidia::cvcore::Bi3D", "featnet_input_layers_name", featnet_input_layers_name_);
  getNitrosContext().setParameter1DStrVector(
    "bi3d_dla0", "nvidia::cvcore::Bi3D", "featnet_output_layers_name", featnet_output_layers_name_);
  getNitrosContext().setParameterStr(
    "bi3d_dla1", "nvidia::cvcore::Bi3D", "featnet_engine_file_path", featnet_engine_file_path_);
  getNitrosContext().setParameter1DStrVector(
    "bi3d_dla1", "nvidia::cvcore::Bi3D", "featnet_input_layers_name", featnet_input_layers_name_);
  getNitrosContext().setParameter1DStrVector(
    "bi3d_dla1", "nvidia::cvcore::Bi3D", "featnet_output_layers_name", featnet_output_layers_name_);

  getNitrosContext().setParameterStr(
    "bi3d_dla0", "nvidia::cvcore::Bi3D", "segnet_engine_file_path", segnet_engine_file_path_);
  getNitrosContext().setParameter1DStrVector(
    "bi3d_dla0", "nvidia::cvcore::Bi3D", "segnet_input_layers_name", segnet_input_layers_name_);
  getNitrosContext().setParameter1DStrVector(
    "bi3d_dla0", "nvidia::cvcore::Bi3D", "segnet_output_layers_name", segnet_output_layers_name_);
  getNitrosContext().setParameterStr(
    "bi3d_dla1", "nvidia::cvcore::Bi3D", "segnet_engine_file_path", segnet_engine_file_path_);
  getNitrosContext().setParameter1DStrVector(
    "bi3d_dla1", "nvidia::cvcore::Bi3D", "segnet_input_layers_name", segnet_input_layers_name_);
  getNitrosContext().setParameter1DStrVector(
    "bi3d_dla1", "nvidia::cvcore::Bi3D", "segnet_output_layers_name", segnet_output_layers_name_);

  // Set max allowed number of disparity values
  getNitrosContext().setParameterUInt64(
    "bi3d_dla0", "nvidia::cvcore::Bi3D", "max_disparity_values", max_disparity_values_);
  getNitrosContext().setParameterUInt64(
    "bi3d_dla1", "nvidia::cvcore::Bi3D", "max_disparity_values", max_disparity_values_);

  // Set Bi3D block memory size depending on maximum number of output disparities
  getNitrosContext().setParameterUInt64(
    "bi3d_dla0", "nvidia::gxf::BlockMemoryPool", "block_size",
    BI3D_BLOCK_SIZE * max_disparity_values_);
  getNitrosContext().setParameterUInt64(
    "bi3d_dla1", "nvidia::gxf::BlockMemoryPool", "block_size",
    BI3D_BLOCK_SIZE * max_disparity_values_);
  getNitrosContext().setParameterUInt64(
    "bi3d_postprocess", "nvidia::gxf::BlockMemoryPool", "block_size",
    BI3D_BLOCK_SIZE * max_disparity_values_);

  RCLCPP_INFO(
    get_logger(), "[Bi3DNode] Setting featnet_engine_file_path: %s.",
    featnet_engine_file_path_.c_str());
  RCLCPP_INFO(
    get_logger(), "[Bi3DNode] Setting segnet_engine_file_path: %s.",
    segnet_engine_file_path_.c_str());
}

Bi3DNode::~Bi3DNode() {}

}  // namespace bi3d
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::bi3d::Bi3DNode)
