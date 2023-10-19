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
#include "extensions/bi3d/components/bi3d_depthview.hpp"

#include <fstream>
#include <iomanip>
#include <string>

#include "engine/core/image/image.hpp"
#include "engine/core/tensor/tensor.hpp"
#include "engine/gems/image/io.hpp"
#include "engine/gems/image/utils.hpp"
#include "gxf/multimedia/video.hpp"

namespace nvidia {
namespace isaac {
namespace bi3d {

// Indicates Depth View Color Scheme.
enum DepthViewColorScheme {
  DepthViewColorScheme_ErrorColor = 1,
  DepthViewColorScheme_RedGreenBlack,
  DepthViewColorScheme_DistanceMap,
  DepthViewColorScheme_StarryNight,
  DepthViewColorScheme_BlackAndWhite
};

gxf_result_t DepthView::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(
      data_receiver_, "data_receiver",
      "Receiver to get the data", "");
  result &= registrar->parameter(
      output_tensor_, "output_tensor",
      "The name of the tensor to be transmitted", "");
  result &= registrar->parameter<bool>(
      output_as_tensor_, "output_as_tensor",
      "The flag to indicate that output is a tensor or video buffer", "", true);
  result &= registrar->parameter(data_transmitter_, "data_transmitter",
      "Transmitter to send the data", "");
  result &= registrar->parameter(
      allocator_, "allocator",
      "Memory pool for allocating output data", "");
  result &= registrar->parameter<int>(
      color_scheme_, "color_scheme",
      "The color scheme for Bi3d depth",
      "Following are supported color schemes: "
      "1 - ErrorColor"
      "2 - RedGreenBlack"
      "3 - DistanceMap"
      "4 - StarryNight"
      "5 - BlackAndWhite",
      5);
  result &= registrar->parameter<float>(
      min_disparity_, "min_disparity",
      "Minimum disparity used for colorization", "0.0f");
  result &= registrar->parameter<float>(
      max_disparity_, "max_disparity",
      "Maximum disparity used for colorization", "255.0f");
  return gxf::ToResultCode(result);
}

gxf_result_t DepthView::start() {
  switch (color_scheme_) {
    case DepthViewColorScheme_ErrorColor:
      gradient_ = ::nvidia::isaac::ErrorColorGradient();
      break;

    case DepthViewColorScheme_RedGreenBlack:
      gradient_ = ::nvidia::isaac::ColorGradient({
        ::nvidia::isaac::Pixel3ub{0, 0, 0}, ::nvidia::isaac::Pixel3ub{118, 185,   0},
        ::nvidia::isaac::Pixel3ub{221, 86, 47}, ::nvidia::isaac::Pixel3ub{255, 0, 0}});
      break;

    case DepthViewColorScheme_DistanceMap:
      gradient_ = ::nvidia::isaac::DistanceColorGradient();
      break;

    case DepthViewColorScheme_StarryNight:
      gradient_ = ::nvidia::isaac::StarryNightColorGradient();
      break;

    case DepthViewColorScheme_BlackAndWhite:
      gradient_ = ::nvidia::isaac::BlackWhiteColorGradient();
      break;

    default:
      gradient_ = ::nvidia::isaac::BlackWhiteColorGradient();
      break;
  }
  return GXF_SUCCESS;
}

gxf_result_t DepthView::tick() {
  auto entity = data_receiver_->receive();
  if (!entity) {
    return gxf::ToResultCode(entity);
  }

  // Get input
  auto input_frame = entity.value().get<gxf::VideoBuffer>();
  if (!input_frame) {
    GXF_LOG_ERROR("Failed to get input frame from message");
    return gxf::ToResultCode(input_frame);
  }

  if (input_frame.value()->video_frame_info().color_format !=
      gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F) {
    GXF_LOG_ERROR("Only supports D32F images");
    return GXF_FAILURE;
  }

  if (input_frame.value()->storage_type() != gxf::MemoryStorageType::kHost) {
    GXF_LOG_ERROR("Only supports video buffer on Host.");
    return GXF_FAILURE;
  }

  int inputWidth  = input_frame.value()->video_frame_info().width;
  int inputHeight = input_frame.value()->video_frame_info().height;
  ::nvidia::isaac::ImageView1f input_depthview = ::nvidia::isaac::CreateImageView<float, 1>(
      reinterpret_cast<float*>(input_frame.value()->pointer()),
      inputHeight, inputWidth);

  // Create output message
  auto output_message = gxf::Entity::New(context());
  if (!output_message) {
    return gxf::ToResultCode(output_message);
  }

  auto output_name = output_tensor_.try_get() ? output_tensor_.try_get()->c_str() : nullptr;
  ::nvidia::isaac::ImageView3ub output_image;

  if (output_as_tensor_) {
    // Create tensor
    auto output_tensor = output_message.value().add<gxf::Tensor>(output_name);
    if (!output_tensor) {
      return gxf::ToResultCode(output_tensor);
    }

    const gxf::Shape shape({inputHeight, inputWidth, 3});
    auto result =
      output_tensor.value()->reshapeCustom(shape, gxf::PrimitiveType::kUnsigned8, 1,
                                           gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
                                           gxf::MemoryStorageType::kHost, allocator_);
    if (!result) {
      GXF_LOG_ERROR("reshape tensor failed.");
      return GXF_FAILURE;
    }

    output_image = ::nvidia::isaac::CreateImageView<uint8_t, 3>(
        output_tensor.value()->data<uint8_t>().value(),
        inputHeight, inputWidth);
  } else {
    // Create video buffer
    auto output_frame = output_message.value().add<gxf::VideoBuffer>(output_name);
    if (!output_frame) {
      return gxf::ToResultCode(output_frame);
    }

    std::array<gxf::ColorPlane, 1> planes{
        gxf::ColorPlane("RGB", 3, inputWidth * 3)};
    gxf::VideoFormatSize<gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB> video_format_size;
    const uint64_t size = video_format_size.size(inputWidth, inputHeight, planes);
    const gxf::VideoBufferInfo buffer_info{static_cast<uint32_t>(inputWidth),
                                           static_cast<uint32_t>(inputHeight),
                                           gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB,
                                           {planes.begin(), planes.end()},
                                           gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR};
    output_frame.value()->resizeCustom(
        buffer_info, size,
        gxf::MemoryStorageType::kHost, allocator_);
    output_image = ::nvidia::isaac::CreateImageView<uint8_t, 3>(
        output_frame.value()->pointer(),
        inputHeight, inputWidth);
  }

  ::nvidia::isaac::Colorize(input_depthview, gradient_, min_disparity_,
                            max_disparity_, output_image);

  // Send the data
  return gxf::ToResultCode(data_transmitter_->publish(output_message.value()));
}

}  // namespace bi3d
}  // namespace isaac
}  // namespace nvidia
