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

#ifndef NVIDIA_CVCORE_BI3D_HPP
#define NVIDIA_CVCORE_BI3D_HPP

#include <cstring>
#include <fstream>
#include <vector>

#include <cv/bi3d/Bi3D.h>

#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace cvcore {

// CV-Core Bi3D GXF Codelet
class Bi3D : public gxf::Codelet {
public:
  Bi3D()  = default;
  ~Bi3D() = default;

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t initialize() override {
    return GXF_SUCCESS;
  }
  gxf_result_t deinitialize() override {
    return GXF_SUCCESS;
  }

  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

private:
  // gxf stream handling
  gxf::Handle<gxf::CudaStream> cuda_stream_ = nullptr;

  // cvcore image pre-processing params for Bi3D
  ::cvcore::ImagePreProcessingParams preProcessorParams;
  // cvcore model input params for Bi3D
  ::cvcore::ModelInputParams modelInputParams;
  // cvcore inference params for Bi3D
  ::cvcore::inferencer::TensorRTInferenceParams featureInferenceParams;
  ::cvcore::inferencer::TensorRTInferenceParams featureHRInferenceParams;
  ::cvcore::inferencer::TensorRTInferenceParams refinementInferenceParams;
  ::cvcore::inferencer::TensorRTInferenceParams segmentationInferenceParams;
  // extra params for Bi3D
  ::cvcore::bi3d::Bi3D::Bi3DParams extraParams;
  // cvcore Bi3D object
  std::unique_ptr<::cvcore::bi3d::Bi3D> objBi3D;
  ::cvcore::Array<int> disparityValues;

  // The name of the input left image tensor
  gxf::Parameter<std::string> left_image_name_;
  // The name of the input right image tensor
  gxf::Parameter<std::string> right_image_name_;
  // The name of the output tensor
  gxf::Parameter<std::string> output_name_;
  // Data allocator to create a tensor
  gxf::Parameter<gxf::Handle<gxf::CudaStreamPool>> stream_pool_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> pool_;
  // Data receiver to get left image data
  gxf::Parameter<gxf::Handle<gxf::Receiver>> left_image_receiver_;
  // Data receiver to get right image data
  gxf::Parameter<gxf::Handle<gxf::Receiver>> right_image_receiver_;
  // Optional receiver for dynamic disparity values input
  gxf::Parameter<gxf::Handle<gxf::Receiver>> disparity_receiver_;
  // Data transmitter to send the data
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> output_transmitter_;

  // Pre-processing params for Bi3D
  gxf::Parameter<std::string> image_type_;
  gxf::Parameter<std::vector<float>> pixel_mean_;
  gxf::Parameter<std::vector<float>> normalization_;
  gxf::Parameter<std::vector<float>> standard_deviation_;

  // Model input params for Bi3D
  gxf::Parameter<int> max_batch_size_;
  gxf::Parameter<int> input_layer_width_;
  gxf::Parameter<int> input_layer_height_;
  gxf::Parameter<std::string> model_input_type_;

  // Inference params for Bi3D
  gxf::Parameter<std::string> featnet_engine_file_path_;
  gxf::Parameter<std::vector<std::string>> featnet_input_layers_name_;
  gxf::Parameter<std::vector<std::string>> featnet_output_layers_name_;

  gxf::Parameter<std::string> featnet_hr_engine_file_path_;
  gxf::Parameter<std::vector<std::string>> featnet_hr_input_layers_name_;
  gxf::Parameter<std::vector<std::string>> featnet_hr_output_layers_name_;

  gxf::Parameter<std::string> refinenet_engine_file_path_;
  gxf::Parameter<std::vector<std::string>> refinenet_input_layers_name_;
  gxf::Parameter<std::vector<std::string>> refinenet_output_layers_name_;

  gxf::Parameter<std::string> segnet_engine_file_path_;
  gxf::Parameter<std::vector<std::string>> segnet_input_layers_name_;
  gxf::Parameter<std::vector<std::string>> segnet_output_layers_name_;

  // Extra params for Bi3D
  gxf::Parameter<std::string> engine_type_;
  gxf::Parameter<bool> apply_sigmoid_;
  gxf::Parameter<bool> apply_thresholding_;
  gxf::Parameter<float> threshold_value_low_;
  gxf::Parameter<float> threshold_value_high_;
  gxf::Parameter<float> threshold_;
  gxf::Parameter<bool> apply_edge_refinement_;
  gxf::Parameter<size_t> max_disparity_levels_;
  gxf::Parameter<std::vector<int>> disparity_values_;

  // Decide which timestamp to pass down
  gxf::Parameter<int> timestamp_policy_;
};

} // namespace cvcore
} // namespace nvidia

#endif
