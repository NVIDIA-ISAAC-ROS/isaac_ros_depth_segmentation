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
#include <string>

#include "gxf/core/gxf.h"
#include "gxf/std/extension_factory_helper.hpp"
#include "bi3d_postprocess/bi3d_postprocessor.hpp"
#include "bi3d_postprocess/bi3d_message_splitter.hpp"

extern "C" {

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(
  0xb764a9fce34d11ec, 0x8fea0242ac120002, "Bi3DPostprocessorExtension",
  "Bi3D Post Processing GXF extension",
  "NVIDIA", "1.0.0", "LICENSE");

GXF_EXT_FACTORY_ADD(
  0xa8a24ecde2544d6a, 0x9a4b81cdb71085d6,
  nvidia::isaac_ros::Bi3DPostprocessor, nvidia::gxf::Codelet,
  "Collapses Bi3D output tensorlist to single segmentation image");

GXF_EXT_FACTORY_ADD(
  0xa8a24ecde2564d6a, 0x9a4b89cdb71085d6,
  nvidia::isaac_ros::Bi3DMessageSplitter, nvidia::gxf::Codelet,
  "Splits Bi3D postprocessor output into disparity image and disparity values");

GXF_EXT_FACTORY_END()

}  // extern "C"
