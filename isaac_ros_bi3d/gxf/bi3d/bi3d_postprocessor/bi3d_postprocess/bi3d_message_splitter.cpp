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
#include "gxf/multimedia/video.hpp"
#include "bi3d_message_splitter.hpp"

namespace nvidia
{
namespace isaac_ros
{

gxf_result_t Bi3DMessageSplitter::registerInterface(gxf::Registrar * registrar) noexcept
{
  gxf::Expected<void> result;
  result &= registrar->parameter(
    receiver_, "receiver", "Receiver",
    "Message from Bi3D post processor");
  result &= registrar->parameter(
    disparity_image_transmitter_, "disparity_image_transmitter", "Disparity image transmitter",
    "The collapsed output of Bi3D");
  result &= registrar->parameter(
    disparity_values_transmitter_, "disparity_values_transmitter", "Disparity values transmitter",
    "The disparity values used for Bi3D inference");
  return gxf::ToResultCode(result);
}

gxf_result_t Bi3DMessageSplitter::tick() noexcept
{
  // Receive bi3d post processor output
  auto bi3d_postprocessed_message = receiver_->receive();
  if (!bi3d_postprocessed_message) {
    GXF_LOG_ERROR("Failed to get message");
    return gxf::ToResultCode(bi3d_postprocessed_message);
  }

  // Publish message
  auto result = disparity_image_transmitter_->publish(bi3d_postprocessed_message.value());
  if (!result) {
    return gxf::ToResultCode(result);
  }
  result = disparity_values_transmitter_->publish(bi3d_postprocessed_message.value());
  if (!result) {
    return gxf::ToResultCode(result);
  }

  return GXF_SUCCESS;
  
}

}  // namespace isaac_ros
}  // namespace nvidia
