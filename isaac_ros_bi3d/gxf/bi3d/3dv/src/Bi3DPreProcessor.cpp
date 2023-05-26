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

struct Bi3DPreProcessor::Bi3DPreProcessorImpl 
{
    mutable detail::InputImage m_leftImageDevice;
    mutable detail::InputImage m_rightImageDevice;
    mutable detail::PreprocessedImage m_preLeftImageDevice;
    mutable detail::PreprocessedImage m_preRightImageDevice;
    mutable detail::InputImage m_leftImageResizeDevice;
    mutable detail::InputImage m_rightImageResizeDevice;

    // Operation(s)
    detail::NormalizeFunctor m_normalizer;

    // Needed parameters for pre-processing
    ImagePreProcessingParams m_preProcessorParams;
    ModelInputParams m_modelParams;

    Bi3DPreProcessorImpl(const ImagePreProcessingParams & preProcessorParams, 
                         const ModelInputParams & modelInputParams,
                         cudaStream_t stream)
        : m_preProcessorParams{preProcessorParams},
          m_modelParams{modelInputParams},
          m_normalizer{modelInputParams.inputLayerWidth, modelInputParams.inputLayerHeight}
    {
        // Configuring the top-level model input(s)
        m_leftImageResizeDevice = {modelInputParams.inputLayerWidth,
                                   modelInputParams.inputLayerHeight,
                                   false};
        m_rightImageResizeDevice = {modelInputParams.inputLayerWidth,
                                    modelInputParams.inputLayerHeight,
                                    m_leftImageResizeDevice.isCPU()};
    }

    void resizeBuffers(std::size_t width, std::size_t height)
    {
        if(m_leftImageDevice.getDimCount() > 0 &&
           m_leftImageDevice.getWidth() == width &&
           m_leftImageDevice.getHeight() == height)
        {
            return;
        }

        m_leftImageDevice = {width, height, false};
        m_rightImageDevice = {width, height, false};
        m_preLeftImageDevice = {width, height, false};
        m_preRightImageDevice = {width, height, false};
    }
};

// =============================================================================
// Bi3D Frontend
// =============================================================================

Bi3DPreProcessor::Bi3DPreProcessor(const ImagePreProcessingParams & preProcessorParams, 
                                   const ModelInputParams & modelInputParams,
                                   cudaStream_t stream)
  : m_pImpl{new Bi3DPreProcessorImpl(preProcessorParams, modelInputParams, stream)} {}

Bi3DPreProcessor::~Bi3DPreProcessor() {}

void Bi3DPreProcessor::execute(detail::PreprocessedImage & preprocessedLeftImage,
                               detail::PreprocessedImage & preprocessedRightImage,
                               const detail::InputImage & leftImage,
                               const detail::InputImage & rightImage,
                               cudaStream_t stream)
{
#ifdef NVBENCH_ENABLE
    const std::string testName = "Bi3DPreProcessor_batch" +
                                 std::to_string(m_pImpl->m_modelParams.maxBatchSize) + "_" +
                                 std::to_string(leftImage.getWidth()) + "x" +
                                 std::to_string(leftImage.getHeight()) + "_GPU";
    nv::bench::Timer timerFunc = nv::bench::GPU(testName.c_str(), nv::bench::Flag::DEFAULT, stream);
#endif

    // Ensuring the buffers are appropreately allocated
    m_pImpl->resizeBuffers(leftImage.getWidth(), leftImage.getHeight());

    // Ensuring data is on the GPU
    // FIXME: Rename so this is not confused with cvcore::Copy
    Copy(m_pImpl->m_leftImageDevice, leftImage, stream);
    Copy(m_pImpl->m_rightImageDevice, rightImage, stream);

    // Resizing the data to model input size
    tensor_ops::Resize(m_pImpl->m_leftImageResizeDevice, m_pImpl->m_leftImageDevice,
                       false, tensor_ops::INTERP_LINEAR, stream);
    tensor_ops::Resize(m_pImpl->m_rightImageResizeDevice, m_pImpl->m_rightImageDevice,
                       false, tensor_ops::INTERP_LINEAR, stream);

    // Normalize (data whiten) the images
    m_pImpl->m_normalizer(preprocessedLeftImage, m_pImpl->m_leftImageResizeDevice,
                          m_pImpl->m_preProcessorParams.pixelMean,
                          m_pImpl->m_preProcessorParams.stdDev,
                          stream);
    m_pImpl->m_normalizer(preprocessedRightImage, m_pImpl->m_rightImageResizeDevice,
                          m_pImpl->m_preProcessorParams.pixelMean,
                          m_pImpl->m_preProcessorParams.stdDev,
                          stream);
}

void Bi3DPreProcessor::execute(detail::PreprocessedImage & preprocessedLeftImage,
                               detail::PreprocessedImage & preprocessedRightImage,
                               const detail::PreprocessedImage & leftImage,
                               const detail::PreprocessedImage & rightImage,
                               cudaStream_t stream)
{
#ifdef NVBENCH_ENABLE
    const std::string testName = "Bi3DPreProcessor_batch" +
                                 std::to_string(m_pImpl->m_modelParams.maxBatchSize) + "_" +
                                 std::to_string(leftImage.getWidth()) + "x" +
                                 std::to_string(leftImage.getHeight()) + "_GPU";
    nv::bench::Timer timerFunc = nv::bench::GPU(testName.c_str(), nv::bench::Flag::DEFAULT, stream);
#endif

    // Ensuring the buffers are appropreately allocated
    m_pImpl->resizeBuffers(leftImage.getWidth(), leftImage.getHeight());

    // FIXME: Rename so this is not confused with cvcore::Copy
    Copy(m_pImpl->m_preLeftImageDevice, leftImage, stream);
    Copy(m_pImpl->m_preRightImageDevice, rightImage, stream);

    // Resizing the data to model input size
    detail::Resize(preprocessedLeftImage, m_pImpl->m_preLeftImageDevice,
                   tensor_ops::INTERP_LINEAR, stream);
    detail::Resize(preprocessedLeftImage, m_pImpl->m_preRightImageDevice,
                   tensor_ops::INTERP_LINEAR, stream);
}

}} // namespace cvcore::bi3d