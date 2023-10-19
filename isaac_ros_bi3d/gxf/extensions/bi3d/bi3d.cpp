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
#include "extensions/bi3d/components/bi3d_inference.hpp"
#include "extensions/bi3d/components/bi3d_postprocessor.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0xce7c6985267a4ec7, 0xa073030e16e49f29, "Bi3D",
                         "Extension containing Bi3D related components",
                         "Isaac SDK", "2.0.0", "LICENSE");

GXF_EXT_FACTORY_ADD(0x9cfc86f71c7c4102, 0xade57d2d4faf6453,
                    nvidia::isaac::bi3d::DepthView, nvidia::gxf::Codelet,
                    "Bi3d depth map generator");

GXF_EXT_FACTORY_ADD(0x2eb361c045894aec, 0x831550ff5f177d87,
                    nvidia::isaac::bi3d::Bi3DPostprocessor, nvidia::gxf::Codelet,
                    "Bi3D post processor");

GXF_EXT_FACTORY_ADD(0xdcba0cf83a5340d2, 0x8404788350ad2324,
                    nvidia::isaac::Bi3DInference, nvidia::gxf::Codelet,
                    "Bi3D Inference");

GXF_EXT_FACTORY_END()
