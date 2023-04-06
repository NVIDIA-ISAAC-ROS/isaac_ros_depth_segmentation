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

#include <cv/bi3d/Bi3D.h>

#include "Bi3D_detail.h"

#include <cuda_runtime.h>

#ifdef NVBENCH_ENABLE
#include <nvbench/CPU.h>
#include <nvbench/GPU.h>
#endif

#include <cv/core/Memory.h>

namespace cvcore { namespace bi3d {

struct Bi3DPostProcessor::Bi3DPostProcessorImpl
{
    mutable detail::DisparityConfidence m_disparityConfidenceDevice;
    mutable detail::DisparityConfidence m_disparityConfidenceThresholdDevice;
    mutable detail::DisparityConfidence m_disparityConfidenceResizeDevice;

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

    void resizeBuffers(std::size_t width, std::size_t height)
    {
        if(m_disparityConfidenceResizeDevice.getDimCount() > 0 &&
           m_disparityConfidenceResizeDevice.getWidth() == width &&
           m_disparityConfidenceResizeDevice.getHeight() == height)
        {
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

void Bi3DPostProcessor::execute(detail::DisparityConfidence & disparityConfidence,
                                detail::DisparityConfidence & rawDisparityConfidence,
                                cudaStream_t stream)
{
#ifdef NVBENCH_ENABLE
    const std::string testName = "Bi3DPostProcessor_batch" +
                                 std::to_string(m_pImpl->m_modelParams.maxBatchSize) + "_GPU";
    nv::bench::Timer timerFunc = nv::bench::GPU(testName.c_str(), nv::bench::Flag::DEFAULT, stream);
#endif

    // Ensuring the buffers are appropreately allocated
    m_pImpl->resizeBuffers(disparityConfidence.getWidth(), disparityConfidence.getHeight());

#ifdef NVBENCH_ENABLE
    {
    const std::string testName = "Bi3DPostProcessorTransformBlock_batch" +
                                 std::to_string(m_pImpl->m_modelParams.maxBatchSize) + "_GPU";
    nv::bench::Timer timerFunc = nv::bench::GPU(testName.c_str(), nv::bench::Flag::DEFAULT, stream);
#endif
    if(m_pImpl->m_params.sigmoidPostProcessing == ProcessingControl::ENABLE)
    {
        detail::Sigmoid(m_pImpl->m_disparityConfidenceDevice,
                        rawDisparityConfidence, stream);

        if(m_pImpl->m_params.thresholdPostProcessing == ProcessingControl::ENABLE)
        {
            detail::Threshold(m_pImpl->m_disparityConfidenceThresholdDevice,
                              m_pImpl->m_disparityConfidenceDevice,
                              m_pImpl->m_params.thresholdValueLow,
                              m_pImpl->m_params.thresholdValueHigh,
                              m_pImpl->m_params.threshold, stream);
            
            detail::Resize(m_pImpl->m_disparityConfidenceResizeDevice,
                           m_pImpl->m_disparityConfidenceThresholdDevice,
                           tensor_ops::INTERP_LINEAR, stream);
        }
        else
        {
            detail::Resize(m_pImpl->m_disparityConfidenceResizeDevice,
                           m_pImpl->m_disparityConfidenceDevice,
                           tensor_ops::INTERP_LINEAR, stream);
        }
    }
    else
    {
        if(m_pImpl->m_params.thresholdPostProcessing == ProcessingControl::ENABLE)
        {
            detail::Threshold(m_pImpl->m_disparityConfidenceThresholdDevice,
                              rawDisparityConfidence,
                              m_pImpl->m_params.thresholdValueLow,
                              m_pImpl->m_params.thresholdValueHigh,
                              m_pImpl->m_params.threshold, stream);
            
            detail::Resize(m_pImpl->m_disparityConfidenceResizeDevice,
                           m_pImpl->m_disparityConfidenceThresholdDevice,
                           tensor_ops::INTERP_LINEAR, stream);
        }
        else
        {
            detail::Resize(m_pImpl->m_disparityConfidenceResizeDevice,
                           rawDisparityConfidence,
                           tensor_ops::INTERP_LINEAR, stream);
        }
    }
#ifdef NVBENCH_ENABLE
    }
#endif

    detail::DisparityConfidence partial{disparityConfidence.getWidth(),
                                        disparityConfidence.getHeight(),
                                        disparityConfidence.getChannelCount(),
                                        m_pImpl->m_disparityConfidenceResizeDevice.getData(),
                                        m_pImpl->m_disparityConfidenceResizeDevice.isCPU()};
    Copy(disparityConfidence, partial, stream);
}

}} // namespace cvcore::bi3d
