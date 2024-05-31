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
#pragma once

#include <string>
#include <vector>

#include "gems/image/color.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace isaac {
namespace bi3d {

class DepthView : public gxf::Codelet {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override {return GXF_SUCCESS;};

 private:
  // Data receiver to get data
  gxf::Parameter<gxf::Handle<gxf::Receiver>> data_receiver_;
  // The name of the output tensor
  gxf::Parameter<std::string> output_tensor_;
  // The flag that indicates if output as a tensor
  gxf::Parameter<bool> output_as_tensor_;
  // Data transmitter to send the data
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> data_transmitter_;
  // Allocator
  gxf::Parameter<gxf::Handle<gxf::Allocator>> allocator_;
  // Color scheme
  gxf::Parameter<int> color_scheme_;
  // Minimum disparity value
  gxf::Parameter<float> min_disparity_;
  // maximum disparity value
  gxf::Parameter<float> max_disparity_;

  // Color gradient used to color different depths
  ::nvidia::isaac::ColorGradient gradient_;
};

}  // namespace bi3d
}  // namespace isaac
}  // namespace nvidia
