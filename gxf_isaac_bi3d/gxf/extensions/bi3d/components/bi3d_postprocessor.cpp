// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "extensions/bi3d/components/bi3d_postprocessor.hpp"

#include <string>

#include "extensions/bi3d/components/bi3d_postprocessor.cu.hpp"
#include "gems/video_buffer/allocator.hpp"
#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace isaac {
namespace bi3d {

gxf_result_t Bi3DPostprocessor::registerInterface(gxf::Registrar* registrar) noexcept {
  gxf::Expected<void> result;
  result &= registrar->parameter(
      bi3d_receiver_, "bi3d_receiver", "Bi3D receiver",
      "Tensor from Bi3D");
  result &= registrar->parameter(
      output_transmitter_, "output_transmitter", "Output transmitter",
      "The collapsed output of Bi3D");
  result &= registrar->parameter(
      pool_, "pool", "Pool",
      "Allocator instance for output videobuffer");
  result &= registrar->parameter(
      disparity_values_, "disparity_values", "Disparity values"
      "Disparities values used for Bi3D inference");
  return gxf::ToResultCode(result);
}

gxf_result_t Bi3DPostprocessor::tick() noexcept {
  // Receive bi3d output
  auto bi3d_message = bi3d_receiver_->receive();
  if (!bi3d_message) {
    GXF_LOG_ERROR("Failed to get message");
    return gxf::ToResultCode(bi3d_message);
  }

  // Get Bi3D disparity tensor
  auto bi3d_disparity_tensor = bi3d_message->get<gxf::Tensor>("disparity");
  if (!bi3d_disparity_tensor) {
    GXF_LOG_ERROR("Failed to get disparity image tensor");
    return gxf::ToResultCode(bi3d_disparity_tensor);
  }

  if (!bi3d_disparity_tensor.value()->data<float>()) {
    GXF_LOG_ERROR("Failed to get pointer to bi3d disparity data");
    return gxf::ToResultCode(bi3d_disparity_tensor.value()->data<float>());
  }
  const float* bi3d_disparity_data_ptr = bi3d_disparity_tensor.value()->data<float>().value();

  // Get dimensions of tensor
  gxf::Shape dims = bi3d_disparity_tensor.value()->shape();
  const int image_height = dims.dimension(1);
  const int image_width = dims.dimension(2);
  const int image_size = dims.dimension(1) * dims.dimension(2);

  // Create output message
  auto out_message = gxf::Entity::New(context());
  if (!out_message) {
    GXF_LOG_ERROR("Failed to allocate message");
    return gxf::ToResultCode(out_message);
  }

  // Create video buffer
  auto out_video_buffer = out_message->add<gxf::VideoBuffer>();
  if (!out_video_buffer) {
    GXF_LOG_ERROR("Failed to allocate output video buffer");
    return gxf::ToResultCode(out_video_buffer);
  }

  // Allocate video buffer on device (unpadded d32f)
  auto maybe_allocation = AllocateUnpaddedVideoBuffer<gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F>(
      out_video_buffer.value(), image_width, image_height, gxf::MemoryStorageType::kDevice,
      pool_.get());
  if (!maybe_allocation) {
    GXF_LOG_ERROR("Failed to allocate output video buffer's memory.");
    return gxf::ToResultCode(maybe_allocation);
  }

  cudaMemset(out_video_buffer.value()->pointer(), 0,
             image_size * bi3d_disparity_tensor.value()->bytes_per_element());

  for (uint32_t i = 0; i < disparity_values_.get().size(); i++) {
    cuda_postprocess(bi3d_disparity_data_ptr + (i * image_size),
                     reinterpret_cast<float*>(out_video_buffer.value()->pointer()),
                     disparity_values_.get().at(i), image_height, image_width);
  }

  // Add timestamp to message if it is available from input message
  std::string timestamp_name{"timestamp"};
  auto maybe_bi3d_timestamp = bi3d_message->get<nvidia::gxf::Timestamp>();
  if (maybe_bi3d_timestamp) {
    auto out_timestamp = out_message.value().add<gxf::Timestamp>(timestamp_name.c_str());
    if (!out_timestamp) {
      return GXF_FAILURE;
    }
    *out_timestamp.value() = *maybe_bi3d_timestamp.value();
  }

  // Publish message
  auto result = output_transmitter_->publish(out_message.value());
  if (!result) {
    return gxf::ToResultCode(result);
  }

  return GXF_SUCCESS;
}

}  // namespace bi3d
}  // namespace isaac
}  // namespace nvidia
