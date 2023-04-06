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

#include "gxf/std/parameter_parser_std.hpp"
#include "gxf/std/timestamp.hpp"
#include "bi3d_postprocessor.hpp"
#include "bi3d_postprocessor_utils.hpp"
#include "bi3d_postprocessor.cu.hpp"


namespace nvidia
{
namespace isaac_ros
{

gxf_result_t Bi3DPostprocessor::registerInterface(gxf::Registrar * registrar) noexcept
{
  gxf::Expected<void> result;
  result &= registrar->parameter(
    bi3d_receiver_, "bi3d_receiver", "Bi3D input",
    "Tensor from Bi3D");
  result &= registrar->parameter(
    output_transmitter_, "output_transmitter", "Output transmitter",
    "The collapsed output of Bi3D");
  result &= registrar->parameter(
    pool_, "pool", "Pool",
    "Allocator instance for output videobuffer");
  result &= registrar->parameter(
    disparity_values_, "disparity_values", "Disparity values",
    "Fixed disparity values used for Bi3D inference",
    gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
    disparity_tensor_name_, "disparity_tensor_name", "Disparity tensor name",
    "Name of the disparity tensor from Bi3D");
  result &= registrar->parameter(
    disparity_values_tensor_name_, "disparity_values_tensor_name", "Disparity values tensor name",
    "Name of the (dynamic) disparity values tensor from Bi3D",
    gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  return gxf::ToResultCode(result);
}

gxf_result_t Bi3DPostprocessor::start() noexcept
{
  auto fixed_disparity_values = disparity_values_.try_get();
  auto disparity_values_name = disparity_values_tensor_name_.try_get();
  if (fixed_disparity_values && disparity_values_name) {
    GXF_LOG_WARNING(
      "Both fixed disparity values and dynamic disparity values given. Fixed disparity values will be used.");
  }
  return GXF_SUCCESS;
}

gxf_result_t Bi3DPostprocessor::tick() noexcept
{
  // Receive bi3d output
  auto bi3d_message = bi3d_receiver_->receive();
  if (!bi3d_message) {
    GXF_LOG_ERROR("Failed to get message");
    return gxf::ToResultCode(bi3d_message);
  }

  // Create output message
  auto out_message = gxf::Entity::New(context());
  if (!out_message) {
    GXF_LOG_ERROR("Failed to allocate message");
    return gxf::ToResultCode(out_message);
  }

  // Get Bi3D disparity tensor
  auto bi3d_disparity_tensor = bi3d_message->get<gxf::Tensor>(disparity_tensor_name_.get().c_str());
  if (!bi3d_disparity_tensor) {
    GXF_LOG_ERROR("Failed to get disparity image tensor");
    return gxf::ToResultCode(bi3d_disparity_tensor);
  }
  if (!bi3d_disparity_tensor.value()->data<float>()) {
    GXF_LOG_ERROR("Failed to get pointer to bi3d disparity data");
    return gxf::ToResultCode(bi3d_disparity_tensor.value()->data<float>());
  }
  const float * bi3d_disparity_data_ptr = bi3d_disparity_tensor.value()->data<float>().value();


  // Get disparity values used for Bi3D inference
  int disparity_count;
  const int * disparity_values_data;

  auto fixed_disparity_values = disparity_values_.try_get();
  if (fixed_disparity_values) {
    disparity_count = fixed_disparity_values.value().size();
    disparity_values_data = fixed_disparity_values.value().data();
  } else {
    auto disparity_values_name = disparity_values_tensor_name_.try_get();
    if (!disparity_values_name) {
      GXF_LOG_ERROR("Neither dynamic nor fixed disparity values specified");
      return GXF_FAILURE;
    }
    auto bi3d_disparity_value_tensor = bi3d_message->get<gxf::Tensor>(
      disparity_values_name.value().c_str());
    if (!bi3d_disparity_value_tensor) {
      GXF_LOG_ERROR("Failed to get disparity values tensor");
      return gxf::ToResultCode(bi3d_disparity_value_tensor);
    }
    disparity_count = bi3d_disparity_value_tensor.value()->element_count();
    disparity_values_data = bi3d_disparity_value_tensor.value()->data<int>().value();

    // Add dynamic disparity value tensor to output message
    auto forwarded_disp_values = out_message.value().add<gxf::Tensor>(
      bi3d_disparity_value_tensor->name());
    *forwarded_disp_values.value() = std::move(*bi3d_disparity_value_tensor.value());
  }

  // Get dimensions of tensor
  gxf::Shape dims = bi3d_disparity_tensor.value()->shape();
  const int image_height = dims.dimension(1);
  const int image_width = dims.dimension(2);
  const int image_size = dims.dimension(1) * dims.dimension(2);

  // Create video buffer
  auto out_video_buffer = out_message.value().add<gxf::VideoBuffer>();
  if (!out_video_buffer) {
    GXF_LOG_ERROR("Failed to allocate output video buffer");
    return gxf::ToResultCode(out_video_buffer);
  }

  // Allocate video buffer on device (unpadded gray32)
  auto maybe_allocation = AllocateVideoBuffer<gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY32>(
    out_video_buffer.value(), image_width, image_height, gxf::MemoryStorageType::kDevice,
    pool_.get());
  if (!maybe_allocation) {
    GXF_LOG_ERROR("Failed to allocate output video buffer's memory.");
    return gxf::ToResultCode(maybe_allocation);
  }

  cudaMemset(
    out_video_buffer.value()->pointer(), 0,
    image_size * bi3d_disparity_tensor.value()->bytes_per_element());

  for (int i = 0; i < disparity_count; i++) {
    cuda_postprocess(
      bi3d_disparity_data_ptr + (i * image_size),
      (float *)out_video_buffer.value()->pointer(), disparity_values_data[i], image_height,
      image_width);
  }

  // Add timestamp
  std::string timestamp_name{"timestamp"};
  auto maybe_bi3d_timestamp = bi3d_message->get<nvidia::gxf::Timestamp>();
  if (!maybe_bi3d_timestamp) {
    GXF_LOG_ERROR("Failed to get a timestamp from Bi3D output");
  }
  auto out_timestamp = out_message.value().add<gxf::Timestamp>(timestamp_name.c_str());
  if (!out_timestamp) {return GXF_FAILURE;}
  *out_timestamp.value() = *maybe_bi3d_timestamp.value();

  // Publish message
  auto result = output_transmitter_->publish(out_message.value());
  if (!result) {
    return gxf::ToResultCode(result);
  }

  return GXF_SUCCESS;
}

}  // namespace isaac_ros
}  // namespace nvidia
