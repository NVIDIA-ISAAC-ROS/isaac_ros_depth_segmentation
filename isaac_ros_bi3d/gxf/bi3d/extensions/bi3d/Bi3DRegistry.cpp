// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "extensions/bi3d/Bi3D.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0x62a390bbf9f842fb, 0xba93f93393a27d4a, "NvCvBi3DExtension", "CVCORE Bi3D module",
                         "Nvidia_Gxf", "1.0.1", "LICENSE");

GXF_EXT_FACTORY_ADD(0xdcba0cf83a5340d2, 0x8404788350ad2324, nvidia::cvcore::Bi3D, nvidia::gxf::Codelet,
                    "Bi3D GXF Extension");
GXF_EXT_FACTORY_END()
