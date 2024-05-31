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

#include <cuda_runtime.h>

#include <string>
#include "extensions/bi3d/inference/Bi3D.h"
#include "extensions/bi3d/inference/Bi3D_detail.hpp"
#include "extensions/tensorops/core/Memory.h"

namespace nvidia {
namespace isaac {
namespace bi3d {

struct Bi3DPostProcessor::Bi3DPostProcessorImpl {
    mutable DisparityConfidence m_disparityConfidenceDevice;
    mutable DisparityConfidence m_disparityConfidenceThresholdDevice;
    mutable DisparityConfidence m_disparityConfidenceResizeDevice;

    ModelInputParams m_modelParams;
    Bi3D::Bi3DParams m_params;

    Bi3DPostProcessorImpl(const ModelInputParams & modelInputParams,
                          const Bi3D::Bi3DParams & bi3dParams,
                          cudaStream_t stream)
        : m_modelParams{modelInputParams}, m_params{bi3dParams}
    {
        m_disparityConfidenceDevice = {modelInputParams.inputLayerWidth,
                                       modelInputParams.inputLayerHeight,
                                       bi3dParams.maxDisparityLevels,
                                       false};

        m_disparityConfidenceThresholdDevice = {modelInputParams.inputLayerWidth,
                                                modelInputParams.inputLayerHeight,
                                                bi3dParams.maxDisparityLevels,
                                                false};
    }

    void resizeBuffers(std::size_t width, std::size_t height) {
        if (m_disparityConfidenceResizeDevice.getDimCount() > 0 &&
           m_disparityConfidenceResizeDevice.getWidth() == width &&
           m_disparityConfidenceResizeDevice.getHeight() == height) {
            return;
        }

        m_disparityConfidenceResizeDevice = {width, height,
                                             m_disparityConfidenceDevice.getChannelCount(),
                                             false};
    }
};

// =============================================================================
// Bi3D Frontend
// =============================================================================

Bi3DPostProcessor::Bi3DPostProcessor(const ModelInputParams & modelInputParams,
                                     const Bi3D::Bi3DParams & bi3dParams,
                                     cudaStream_t stream)
  : m_pImpl{new Bi3DPostProcessorImpl(modelInputParams, bi3dParams, stream)} {}

Bi3DPostProcessor::~Bi3DPostProcessor() {}

void Bi3DPostProcessor::execute(DisparityConfidence & disparityConfidence,
                                DisparityConfidence & rawDisparityConfidence,
                                cudaStream_t stream) {
    // Ensuring the buffers are appropreately allocated
    m_pImpl->resizeBuffers(disparityConfidence.getWidth(), disparityConfidence.getHeight());
    if (m_pImpl->m_params.sigmoidPostProcessing == ProcessingControl::ENABLE) {
        Sigmoid(m_pImpl->m_disparityConfidenceDevice,
                        rawDisparityConfidence, stream);

        if (m_pImpl->m_params.thresholdPostProcessing == ProcessingControl::ENABLE) {
            Threshold(m_pImpl->m_disparityConfidenceThresholdDevice,
                              m_pImpl->m_disparityConfidenceDevice,
                              m_pImpl->m_params.thresholdValueLow,
                              m_pImpl->m_params.thresholdValueHigh,
                              m_pImpl->m_params.threshold, stream);

            Resize(m_pImpl->m_disparityConfidenceResizeDevice,
                           m_pImpl->m_disparityConfidenceThresholdDevice,
                           cvcore::tensor_ops::InterpolationType::INTERP_LINEAR, stream);
        } else {
            Resize(m_pImpl->m_disparityConfidenceResizeDevice,
                           m_pImpl->m_disparityConfidenceDevice,
                           cvcore::tensor_ops::InterpolationType::INTERP_LINEAR, stream);
        }
    } else {
        if (m_pImpl->m_params.thresholdPostProcessing == ProcessingControl::ENABLE) {
            Threshold(m_pImpl->m_disparityConfidenceThresholdDevice,
                              rawDisparityConfidence,
                              m_pImpl->m_params.thresholdValueLow,
                              m_pImpl->m_params.thresholdValueHigh,
                              m_pImpl->m_params.threshold, stream);

            Resize(m_pImpl->m_disparityConfidenceResizeDevice,
                           m_pImpl->m_disparityConfidenceThresholdDevice,
                           cvcore::tensor_ops::InterpolationType::INTERP_LINEAR, stream);
        } else {
            Resize(m_pImpl->m_disparityConfidenceResizeDevice,
                           rawDisparityConfidence,
                           cvcore::tensor_ops::InterpolationType::INTERP_LINEAR, stream);
        }
    }
    DisparityConfidence partial{disparityConfidence.getWidth(),
                                        disparityConfidence.getHeight(),
                                        disparityConfidence.getChannelCount(),
                                        m_pImpl->m_disparityConfidenceResizeDevice.getData(),
                                        m_pImpl->m_disparityConfidenceResizeDevice.isCPU()};
    Copy(disparityConfidence, partial, stream);
}
}  // namespace bi3d
}  // namespace isaac
}  // namespace nvidia
