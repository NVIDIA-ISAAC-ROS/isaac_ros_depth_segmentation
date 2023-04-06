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

#include <cuda_runtime.h>

#include <string>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wpedantic"
#include "gxf/core/entity.hpp"
#include "gxf/core/gxf.h"
#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"
#pragma GCC diagnostic pop

#include "isaac_ros_nitros_bi3d_inference_param_array_type/nitros_bi3d_inference_param_array.hpp"
#include "isaac_ros_nitros/types/type_adapter_nitros_context.hpp"

#include "rclcpp/rclcpp.hpp"


constexpr char kEntityName[] = "memory_pool";
constexpr char kComponentName[] = "unbounded_allocator";
constexpr char kComponentTypeName[] = "nvidia::gxf::UnboundedAllocator";

void rclcpp::TypeAdapter<
  nvidia::isaac_ros::nitros::NitrosBi3DInferenceParamArray,
  isaac_ros_bi3d_interfaces::msg::Bi3DInferenceParametersArray>::convert_to_ros_message(
  const custom_type & source,
  ros_message_type & destination)
{
  nvidia::isaac_ros::nitros::nvtxRangePushWrapper(
    "NitrosTensorList::convert_to_ros_message",
    nvidia::isaac_ros::nitros::CLR_PURPLE);

  RCLCPP_DEBUG(
    rclcpp::get_logger("NitrosBi3DInferenceParamArray"),
    "[convert_to_ros_message] Conversion started for handle = %ld", source.handle);

  auto msg_entity = nvidia::gxf::Entity::Shared(
    nvidia::isaac_ros::nitros::GetTypeAdapterNitrosContext().getContext(), source.handle);
  auto gxf_tensor = msg_entity->get<nvidia::gxf::Tensor>();

  if (!gxf_tensor) {
    std::string error_msg =
      "[convert_to_ros_message] No tensor found: conversion from "
      "gxf::Tensor to ROS Bi3D Inference Parameters Array failed!";
    RCLCPP_ERROR(
      rclcpp::get_logger("NitrosBi3DInferenceParamArray"), error_msg.c_str());
    throw std::runtime_error(error_msg.c_str());
  }

  // Move data from GXF tensor to ROS message
  destination.disparity_values.resize(gxf_tensor.value()->size());
  switch (gxf_tensor.value()->storage_type()) {
    case nvidia::gxf::MemoryStorageType::kHost:
      {
        std::memcpy(
          destination.disparity_values.data(), gxf_tensor.value()->pointer(),
          gxf_tensor.value()->size());
      }
      break;
    case nvidia::gxf::MemoryStorageType::kDevice:
      {
        const cudaError_t cuda_error = cudaMemcpy(
          destination.disparity_values.data(), gxf_tensor.value()->pointer(),
          gxf_tensor.value()->size(), cudaMemcpyDeviceToHost);
        if (cuda_error != cudaSuccess) {
          std::stringstream error_msg;
          error_msg <<
            "[convert_to_ros_message] cudaMemcpy failed for conversion from "
            "gxf::Tensor to ROS Bi3D Inference Parameters Array: " <<
            cudaGetErrorName(cuda_error) <<
            " (" << cudaGetErrorString(cuda_error) << ")";
          RCLCPP_ERROR(
            rclcpp::get_logger("NitrosBi3DInferenceParamArray"), error_msg.str().c_str());
          throw std::runtime_error(error_msg.str().c_str());
        }
      }
      break;
    default:
      std::string error_msg =
        "[convert_to_ros_message] MemoryStorageType not supported: conversion from "
        "gxf::Tensor to ROS Bi3D Inference Parameters Array failed!";
      RCLCPP_ERROR(
        rclcpp::get_logger("NitrosBi3DInferenceParamArray"), error_msg.c_str());
      throw std::runtime_error(error_msg.c_str());
  }

  // Populate timestamp information back into ROS header
  auto input_timestamp = msg_entity->get<nvidia::gxf::Timestamp>("timestamp");
  if (!input_timestamp) {      // Fallback to label 'timestamp'
    input_timestamp = msg_entity->get<nvidia::gxf::Timestamp>();
  }
  if (input_timestamp) {
    destination.header.stamp.sec = static_cast<int32_t>(
      input_timestamp.value()->acqtime / static_cast<uint64_t>(1e9));
    destination.header.stamp.nanosec = static_cast<uint32_t>(
      input_timestamp.value()->acqtime % static_cast<uint64_t>(1e9));
  }

  // Set frame ID
  destination.header.frame_id = source.frame_id;

  RCLCPP_DEBUG(
    rclcpp::get_logger("NitrosBi3DInferenceParamArray"),
    "[convert_to_ros_message] Conversion completed");

  nvidia::isaac_ros::nitros::nvtxRangePopWrapper();
}


void rclcpp::TypeAdapter<
  nvidia::isaac_ros::nitros::NitrosBi3DInferenceParamArray,
  isaac_ros_bi3d_interfaces::msg::Bi3DInferenceParametersArray>::convert_to_custom(
  const ros_message_type & source, custom_type & destination)
{
  nvidia::isaac_ros::nitros::nvtxRangePushWrapper(
    "NitrosBi3DInferenceParamArray::convert_to_custom",
    nvidia::isaac_ros::nitros::CLR_PURPLE);

  RCLCPP_DEBUG(
    rclcpp::get_logger("NitrosBi3DInferenceParamArray"),
    "[convert_to_custom] Conversion started");

  // Get pointer to allocator component
  gxf_uid_t cid;
  nvidia::isaac_ros::nitros::GetTypeAdapterNitrosContext().getCid(
    kEntityName, kComponentName, kComponentTypeName, cid);

  auto maybe_allocator_handle =
    nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
    nvidia::isaac_ros::nitros::GetTypeAdapterNitrosContext().getContext(), cid);
  if (!maybe_allocator_handle) {
    std::stringstream error_msg;
    error_msg <<
      "[convert_to_custom] Failed to get allocator's handle: " <<
      GxfResultStr(maybe_allocator_handle.error());
    RCLCPP_ERROR(
      rclcpp::get_logger("NitrosBi3DInferenceParamArray"), error_msg.str().c_str());
    throw std::runtime_error(error_msg.str().c_str());
  }
  auto allocator_handle = maybe_allocator_handle.value();

  auto message = nvidia::gxf::Entity::New(
    nvidia::isaac_ros::nitros::GetTypeAdapterNitrosContext().getContext());
  if (!message) {
    std::stringstream error_msg;
    error_msg <<
      "[convert_to_custom] Error initializing new message entity: " <<
      GxfResultStr(message.error());
    RCLCPP_ERROR(
      rclcpp::get_logger("NitrosBi3DInferenceParamArray"), error_msg.str().c_str());
    throw std::runtime_error(error_msg.str().c_str());
  }

  auto gxf_tensor = message->add<nvidia::gxf::Tensor>("bi3d_inference_disparities");
  std::array<int32_t,
    nvidia::gxf::Shape::kMaxRank> dims = {static_cast<int32_t>(source.disparity_values.size())};

  nvidia::gxf::Expected<void> result;

  // Bi3D disparity value tensor must be on host
  nvidia::gxf::MemoryStorageType storage_type = nvidia::gxf::MemoryStorageType::kHost;

  // Initializing GXF tensor
  nvidia::gxf::PrimitiveType type =
    static_cast<nvidia::gxf::PrimitiveType>(nvidia::gxf::PrimitiveType::kInt32);
  result = gxf_tensor.value()->reshape<int32_t>(
    nvidia::gxf::Shape(dims, 1), storage_type, allocator_handle);

  if (!result) {
    std::stringstream error_msg;
    error_msg <<
      "[convert_to_custom] Error initializing GXF tensor of type " <<
      static_cast<int>(type) << ": " <<
      GxfResultStr(result.error());
    RCLCPP_ERROR(
      rclcpp::get_logger("NitrosBi3DInferenceParamArray"), error_msg.str().c_str());
    throw std::runtime_error(error_msg.str().c_str());
  }

  // Sort disparity values data
  std::vector<int32_t> sorted_data(begin(source.disparity_values), end(source.disparity_values));
  sort(sorted_data.begin(), sorted_data.end());

  // Copy to GXF tensor
  const cudaMemcpyKind operation = cudaMemcpyHostToHost;
  const cudaError_t cuda_error = cudaMemcpy(
    gxf_tensor.value()->pointer(),
    sorted_data.data(),
    gxf_tensor.value()->size(),
    operation);
  if (cuda_error != cudaSuccess) {
    std::stringstream error_msg;
    error_msg <<
      "[convert_to_custom] cudaMemcpy failed for copying data from "
      "Bi3D ROS Float Array to GXF Tensor: " <<
      cudaGetErrorName(cuda_error) <<
      " (" << cudaGetErrorString(cuda_error) << ")";
    RCLCPP_ERROR(
      rclcpp::get_logger("NitrosBi3DInferenceParamArray"), error_msg.str().c_str());
    throw std::runtime_error(error_msg.str().c_str());
  }

  // Add timestamp to the message
  uint64_t input_timestamp =
    source.header.stamp.sec * static_cast<uint64_t>(1e9) +
    source.header.stamp.nanosec;
  auto output_timestamp = message->add<nvidia::gxf::Timestamp>("timestamp");
  if (!output_timestamp) {
    std::stringstream error_msg;
    error_msg << "[convert_to_custom] Failed to add a timestamp component to message: " <<
      GxfResultStr(output_timestamp.error());
    RCLCPP_ERROR(
      rclcpp::get_logger("NitrosBi3DInferenceParamArray"), error_msg.str().c_str());
    throw std::runtime_error(error_msg.str().c_str());
  }
  output_timestamp.value()->acqtime = input_timestamp;

  // Set frame ID
  destination.frame_id = source.header.frame_id;

  // Set message entity
  destination.handle = message->eid();
  GxfEntityRefCountInc(
    nvidia::isaac_ros::nitros::GetTypeAdapterNitrosContext().getContext(), message->eid());

  RCLCPP_DEBUG(
    rclcpp::get_logger("NitrosBi3DInferenceParamArray"),
    "[convert_to_custom] Conversion completed");

  nvidia::isaac_ros::nitros::nvtxRangePopWrapper();
}
