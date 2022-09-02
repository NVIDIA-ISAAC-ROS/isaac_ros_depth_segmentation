/**
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef ISAAC_ROS_NITROS_BI3D_INFERENCE_PARAM_ARRAY_TYPE__NITROS_BI3D_INFERENCE_PARAM_ARRAY_HPP_
#define ISAAC_ROS_NITROS_BI3D_INFERENCE_PARAM_ARRAY_TYPE__NITROS_BI3D_INFERENCE_PARAM_ARRAY_HPP_
/*
 * Type adaptation for:
 *   Nitros type: NitrosBi3DInferenceParamArray
 *   ROS type:    isaac_ros_bi3d_interfaces::msg::Bi3DInferenceParametersArray
 */

#include <string>

#include "isaac_ros_nitros/types/nitros_format_agent.hpp"
#include "isaac_ros_nitros/types/nitros_type_base.hpp"
#include "isaac_ros_bi3d_interfaces/msg/bi3_d_inference_parameters_array.hpp"


#include "rclcpp/type_adapter.hpp"


namespace nvidia
{
namespace isaac_ros
{
namespace nitros
{

// Type
struct NitrosBi3DInferenceParamArray;

// Formats
struct nitros_bi3d_inference_param_array_t
{
  using MsgT = NitrosBi3DInferenceParamArray;
  static const inline std::string supported_type_name = "nitros_bi3d_inference_param_array";
};

// NITROS data type registration factory
NITROS_TYPE_FACTORY_BEGIN(NitrosBi3DInferenceParamArray)
// Supported data formats
NITROS_FORMAT_FACTORY_BEGIN()
NITROS_FORMAT_ADD(nitros_bi3d_inference_param_array_t)
NITROS_FORMAT_FACTORY_END()
// Required extensions
NITROS_TYPE_EXTENSION_FACTORY_BEGIN()
NITROS_TYPE_EXTENSION_ADD("isaac_ros_nitros", "gxf/std/libgxf_std.so")
NITROS_TYPE_EXTENSION_FACTORY_END()
NITROS_TYPE_FACTORY_END()


}  // namespace nitros
}  // namespace isaac_ros
}  // namespace nvidia


template<>
struct rclcpp::TypeAdapter<
  nvidia::isaac_ros::nitros::NitrosBi3DInferenceParamArray,
  isaac_ros_bi3d_interfaces::msg::Bi3DInferenceParametersArray>
{
  using is_specialized = std::true_type;
  using custom_type = nvidia::isaac_ros::nitros::NitrosBi3DInferenceParamArray;
  using ros_message_type = isaac_ros_bi3d_interfaces::msg::Bi3DInferenceParametersArray;

  static void convert_to_ros_message(
    const custom_type & source,
    ros_message_type & destination);

  static void convert_to_custom(
    const ros_message_type & source,
    custom_type & destination);
};

RCLCPP_USING_CUSTOM_TYPE_AS_ROS_MESSAGE_TYPE(
  nvidia::isaac_ros::nitros::NitrosBi3DInferenceParamArray,
  isaac_ros_bi3d_interfaces::msg::Bi3DInferenceParametersArray);

#endif  // ISAAC_ROS_NITROS_BI3D_INFERENCE_PARAM_ARRAY_TYPE__NITROS_BI3D_INFERENCE_PARAM_ARRAY_HPP_
