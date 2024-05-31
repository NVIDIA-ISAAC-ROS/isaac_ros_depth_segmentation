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

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "extensions/bi3d/inference/Bi3D.h"
#include "extensions/bi3d/inference/Bi3D_detail.hpp"
#include "extensions/tensorops/core/BBox.h"
#include "extensions/tensorops/core/BBoxUtils.h"
#include "extensions/tensorops/core/Image.h"
#include "extensions/tensorops/core/ImageUtils.h"
#include "extensions/tensorops/core/Memory.h"
#include "extensions/tensorops/core/NppUtils.h"
#include "extensions/tensorops/core/Tensor.h"
#include "extensions/tensorops/core/Traits.h"
#include "gems/dnn_inferencer/inferencer/Errors.h"
#include "gems/dnn_inferencer/inferencer/Inferencer.h"

#ifndef ENGINE_DIR
#define ENGINE_DIR "models"
#endif  // ENGINE_DIR
namespace nvidia {
namespace isaac {
namespace bi3d {

// =============================================================================
// Model Parameters
// =============================================================================

const ImagePreProcessingParams Bi3D::defaultPreProcessorParams = {
    cvcore::tensor_ops::ImageType::BGR_U8,          /**< Input Image Type. */
    {1.0/127.5, 1.0/127.5, 1.0/127.5}, /**< Image Mean value per channel (OFFSET). */
    {0.00392156862,
     0.00392156862,
     0.00392156862}, /**< Normalization values per channel (SCALE). */
    {-127.5, -127.5, -127.5}  /**< Standard deviation values per channel. */
};

const ModelInputParams Bi3D::defaultModelInputParams = {
    32,      /**< Max Batch Size */
    960,     /**< Width of the model input */
    576,     /**< Height of the model input */
    cvcore::tensor_ops::ImageType::RGB_U8  /**< Format of the model input */
};

const Bi3D::InferencerParams Bi3D::defaultFeatureParams = {
    TRTInferenceType::TRT_ENGINE,
    nullptr,
    "",
    ENGINE_DIR "/bi3dnet_featnet.engine",
    false,
    cvcore::inferencer::OnnxModelBuildFlag::NONE,
    0,
    1,
    {{"input.1"}},
    {{"97"}}
};

const Bi3D::InferencerParams Bi3D::defaultHRParams = {
    TRTInferenceType::TRT_ENGINE,
    nullptr,
    "",
    ENGINE_DIR "/bi3dnet_featnethr.engine",
    false,
    cvcore::inferencer::OnnxModelBuildFlag::NONE,
    0,
    1,
    {{"input.1"}},
    {{"10"}}
};

const Bi3D::InferencerParams Bi3D::defaultRefinementParams = {
    TRTInferenceType::TRT_ENGINE,
    nullptr,
    "",
    ENGINE_DIR "/bi3dnet_refinenet.engine",
    false,
    cvcore::inferencer::OnnxModelBuildFlag::NONE,
    0,
    1,
    {{"input.1"}},
    {{"6"}}
};

const Bi3D::InferencerParams Bi3D::defaultSegmentationParams = {
    TRTInferenceType::TRT_ENGINE,
    nullptr,
    "",
    ENGINE_DIR "/bi3dnet_segnet.engine",
    false,
    cvcore::inferencer::OnnxModelBuildFlag::NONE,
    0,
    1,
    {{"input.1"}},
    {{"278"}}
};

const Bi3D::Bi3DParams Bi3D::defaultBi3DParams = {
    192/3,
    ProcessingControl::DISABLE,
    ProcessingControl::DISABLE,
    ProcessingControl::ENABLE,
    0.0f,
    1.0f,
    0.5f
};

// =============================================================================
// Bi3D Inferencer(s)
// =============================================================================

template<typename InputTensorType, typename OutputTensorType>
class UnaryInferencer {
    using InferencerFactory = cvcore::inferencer::InferenceBackendFactory;
    using LayerInfo = cvcore::inferencer::LayerInfo;

 public:
        UnaryInferencer() = delete;

        UnaryInferencer(const std::string & name,
                        const Bi3D::InferencerParams & modelParams,
                        cudaStream_t stream = 0)
            : m_name{name}, m_inferencer{nullptr}, m_modelParams{modelParams},
              m_inferencerState{cvcore::tensor_ops::ErrorCode::SUCCESS} {
            this->resetInferencerState();
        }

        ~UnaryInferencer() {
            InferencerFactory::DestroyTensorRTInferenceBackendClient(m_inferencer);
        }

        std::error_code getInferencerState() const {
            return m_inferencerState;
        }

        std::error_code resetInferencerState() {
            if (m_inferencer) {
                m_inferencerState =
                  InferencerFactory::DestroyTensorRTInferenceBackendClient(m_inferencer);
            }

            if (!m_inferencerState) {
                m_inferencerState =
                  InferencerFactory::CreateTensorRTInferenceBackendClient(m_inferencer,
                    m_modelParams);
            }

            if (!m_inferencerState) {
                m_modelMeta = m_inferencer->getModelMetaData();
            }

            return m_inferencerState;
        }

        LayerInfo getInputInfo() const {
            return m_modelMeta.inputLayers.at(m_modelParams.inputLayerNames[0]);
        }

        LayerInfo getOutputInfo() const {
            return m_modelMeta.outputLayers.at(m_modelParams.outputLayerNames[0]);
        }

        std::error_code execute(OutputTensorType & response,
                                const InputTensorType & signal,
                                cudaStream_t stream = 0) {
            if (!m_inferencerState) {
                m_inferencerState = m_inferencer->setCudaStream(stream);
            }

            // Assigning the network buffers.
            if (!m_inferencerState) {
                m_inferencerState = m_inferencer->setInput(dynamic_cast<const TensorBase&>(signal),
                                                          m_modelParams.inputLayerNames[0]);
            }
            if (!m_inferencerState) {
                m_inferencerState = m_inferencer->setOutput(dynamic_cast<TensorBase&>(response),
                                                           m_modelParams.outputLayerNames[0]);
            }

            // Run the inference
            if (!m_inferencerState) {
                m_inferencerState = m_inferencer->infer();
            }

            return m_inferencerState;
        }

 private:
        cvcore::inferencer::InferenceBackendClient m_inferencer;
        std::string m_name;
        Bi3D::InferencerParams m_modelParams;
        cvcore::inferencer::ModelMetaData m_modelMeta;
        std::error_code m_inferencerState;
};

using FeatureExtractor    = UnaryInferencer<PreprocessedImage, FeatureResponse>;
using DisparityDetector   = UnaryInferencer<FeatureResponse,
                                            ConfidenceMap>;
using DisparityClassifier = UnaryInferencer<GuidedConfidence,
                                            RefinedConfidence>;

// =============================================================================
// Bi3D Data Exchange
// =============================================================================

struct Bi3D::Bi3DImpl {
    // Memory buffers
    // NOTE: declared mutable because the data in these buffers must always change
    mutable PreprocessedImage m_preprocessedLeftImageDevice;
    mutable PreprocessedImage m_preprocessedRightImageDevice;
    mutable FeatureResponse m_leftFeaturesDevice;
    mutable FeatureResponse m_rightFeaturesDevice;
    mutable FeatureResponse m_hrFeaturesDevice;
    mutable FeatureResponse m_leftFeaturesPadDevice;
    mutable FeatureResponse m_rightFeaturesPadDevice;
    mutable FeatureResponse m_mergedFeaturesDevice;
    mutable ConfidenceMap m_confidenceMapDevice;
    mutable ConfidenceMap m_confidenceMapResizeDevice;
    mutable GuidedConfidence m_fusedDisparityDevice;
    mutable DisparityConfidence m_disparityConfidenceDevice;

    // Exchange buffers
    mutable QuantizedDisparity m_quantizedDisparityDevice;
    mutable DisparityConfidence m_disparityConfidenceExchangeDevice;

    Bi3DParams m_bi3dParams;

    // Processor(s)
    std::unique_ptr<Bi3DPreProcessor> m_preprocess;
    std::unique_ptr<FeatureExtractor> m_featureInfer;
    std::unique_ptr<FeatureExtractor> m_hrInfer;
    std::unique_ptr<DisparityDetector> m_segmentationInfer;
    std::unique_ptr<DisparityClassifier> m_refinementInfer;
    std::unique_ptr<Bi3DPostProcessor> m_postprocess;

    Bi3DImpl(const ImagePreProcessingParams & preProcessorParams,
             const ModelInputParams & modelInputParams,
             const InferencerParams & featureParams,
             const InferencerParams & hrParams,
             const InferencerParams & refinementParams,
             const InferencerParams & segmentationParams,
             const Bi3DParams & bi3dParams,
             cudaStream_t stream)
        : m_preprocess{new Bi3DPreProcessor{preProcessorParams, modelInputParams, stream}},
          m_featureInfer{new FeatureExtractor{"FeatureNetInferencer", featureParams, stream}},
          m_segmentationInfer{new DisparityDetector{"SegmentationNetInferencer",
              segmentationParams, stream}},
          m_postprocess{new Bi3DPostProcessor{modelInputParams, bi3dParams, stream}},
          m_bi3dParams{bi3dParams} {
        // Configuring the preprocessor outputs
        m_preprocessedLeftImageDevice = {m_featureInfer->getInputInfo().shape[3],
                                         m_featureInfer->getInputInfo().shape[2],
                                         false};
        m_preprocessedRightImageDevice = {m_featureInfer->getInputInfo().shape[3],
                                          m_featureInfer->getInputInfo().shape[2],
                                          m_preprocessedLeftImageDevice.isCPU()};

        // Output SegRefine: Accumulated confidence output at all disparity levels
        m_disparityConfidenceDevice = {m_featureInfer->getInputInfo().shape[3],
                                       m_featureInfer->getInputInfo().shape[2],
                                       bi3dParams.maxDisparityLevels,
                                       m_preprocessedLeftImageDevice.isCPU()};

        // The output FeatureHRNet is concatinated together with the resized
        // output of SegNet to form a guidance map as input into RefineNet
        if (m_bi3dParams.edgeRefinement == ProcessingControl::ENABLE) {
            // Load featnethr and refinenet only necessary
            m_hrInfer.reset(new FeatureExtractor{"FeatureHRNetInferencer", hrParams, stream});
            m_refinementInfer.reset(new DisparityClassifier{"RefinementNetInferencer",
                refinementParams, stream});
            m_fusedDisparityDevice = {
                m_refinementInfer->getInputInfo().shape[3],
                m_refinementInfer->getInputInfo().shape[2],
                m_refinementInfer->getInputInfo().shape[1],
                m_preprocessedLeftImageDevice.isCPU()};
            m_confidenceMapResizeDevice = {
                m_refinementInfer->getInputInfo().shape[3],
                m_refinementInfer->getInputInfo().shape[2],
                m_fusedDisparityDevice.getData() + 0*m_fusedDisparityDevice.getStride(
                    TensorDimension::CHANNEL),
                m_preprocessedLeftImageDevice.isCPU()};
            m_hrFeaturesDevice = {
                m_hrInfer->getOutputInfo().shape[3],
                m_hrInfer->getOutputInfo().shape[2],
                m_hrInfer->getOutputInfo().shape[1],
                m_fusedDisparityDevice.getData()
                    + 1*m_fusedDisparityDevice.getStride(TensorDimension::CHANNEL),
                m_preprocessedLeftImageDevice.isCPU()};
        }

        // Output SegNet: Computed disparity confidence at a given disparity level
        m_confidenceMapDevice = {
            m_segmentationInfer->getOutputInfo().shape[3],
            m_segmentationInfer->getOutputInfo().shape[2],
            m_preprocessedLeftImageDevice.isCPU()};

        // Input SegNet: The concatinated left and right extracted image
        //     features from FeatureNet
        m_mergedFeaturesDevice = {
            m_segmentationInfer->getInputInfo().shape[3],
            m_segmentationInfer->getInputInfo().shape[2],
            m_segmentationInfer->getInputInfo().shape[1],
            m_preprocessedLeftImageDevice.isCPU()};

        // The output extracted right features of FetureNet are padded and shifted
        // horizontally by the given disparity level. The left features are zero
        // padded to the same size as the right features.
        // NOTE: This model assumes the left and right image are horizontally
        //     aligned. That is, the transformation from camera left to camera
        //     right is strictly along the horizontal axis relative to lab frame.
        m_leftFeaturesPadDevice = {
            m_segmentationInfer->getInputInfo().shape[3],
            m_featureInfer->getOutputInfo().shape[2],
            m_featureInfer->getOutputInfo().shape[1],
            m_mergedFeaturesDevice.getData()
                + 0*m_mergedFeaturesDevice.getStride(TensorDimension::CHANNEL),
            m_preprocessedLeftImageDevice.isCPU()};
        m_rightFeaturesPadDevice = {
            m_segmentationInfer->getInputInfo().shape[3],
            m_featureInfer->getOutputInfo().shape[2],
            m_featureInfer->getOutputInfo().shape[1],
            m_mergedFeaturesDevice.getData()
                + m_featureInfer->getOutputInfo().shape[1]
                * m_mergedFeaturesDevice.getStride(TensorDimension::CHANNEL),
            m_preprocessedLeftImageDevice.isCPU()};

        // Output FeatureNet: Feature extraction on the left and right images respectively
        m_leftFeaturesDevice = {
            m_featureInfer->getOutputInfo().shape[3],
            m_featureInfer->getOutputInfo().shape[2],
            m_featureInfer->getOutputInfo().shape[1],
            m_preprocessedLeftImageDevice.isCPU()};
        m_rightFeaturesDevice = {
            m_featureInfer->getOutputInfo().shape[3],
            m_featureInfer->getOutputInfo().shape[2],
            m_featureInfer->getOutputInfo().shape[1],
            m_preprocessedLeftImageDevice.isCPU()};
    }

    void resizeBuffers(std::size_t width, std::size_t height) {
        if (m_quantizedDisparityDevice.getDimCount() > 0 &&
           m_quantizedDisparityDevice.getWidth() == width &&
           m_quantizedDisparityDevice.getHeight() == height) {
            return;
        }

        m_disparityConfidenceExchangeDevice = {
            width, height,
            m_disparityConfidenceDevice.getChannelCount(),
            m_preprocessedLeftImageDevice.isCPU()};
        m_quantizedDisparityDevice = {
            width, height,
            m_disparityConfidenceDevice.getChannelCount(),
            m_preprocessedLeftImageDevice.isCPU()};
    }
};

// =============================================================================
// Bi3D Frontend
// =============================================================================

Bi3D::Bi3D(const ImagePreProcessingParams & preProcessorParams,
           const ModelInputParams & modelInputParams,
           const InferencerParams & featureParams,
           const InferencerParams & hrParams,
           const InferencerParams & refinementParams,
           const InferencerParams & segmentationParams,
           const Bi3DParams & bi3dParams,
           cudaStream_t stream)
    : m_pImpl{new Bi3DImpl{preProcessorParams, modelInputParams, featureParams,
                           hrParams, refinementParams, segmentationParams,
                           bi3dParams, stream}} {}

Bi3D::~Bi3D() {}

void Bi3D::execute(DisparityConfidence & disparityConfidence,
                   const PreprocessedImage & leftImage,
                   const PreprocessedImage & rightImage,
                   const DisparityLevels & disparityLevels,
                   cudaStream_t stream) {
    if (disparityConfidence.isCPU()) {
        throw std::invalid_argument("cvcore::bi3d::Bi3D::execute ==> disparityConfidence "
                                    "was allocated on host (CPU) and not device (GPU)");
    }

    if (leftImage.isCPU() || rightImage.isCPU()) {
        throw std::invalid_argument("cvcore::bi3d::Bi3D::execute ==> leftImage or rightImage "
                                    "was allocated on host (CPU) and not device (GPU)");
    }

    if (leftImage.getChannelCount() != rightImage.getChannelCount()) {
        throw std::invalid_argument("cvcore::bi3d::Bi3D::execute ==> "
                                    "leftImage and rightImage have incompatable channel "
                                    "sizes inconsistent with model configuration parameters");
    }

    if (leftImage.getWidth() != rightImage.getWidth() ||
       leftImage.getHeight() != rightImage.getHeight()) {
        throw std::invalid_argument("cvcore::bi3d::Bi3D::execute ==> "
                                    "leftImage and rightImage have incompatable width and height"
                                    "sizes inconsistent with model configuration parameters");
    }

    // This is a boolean flag to indicate execution of the post processing step in the case that:
    //   1) One of the post processing types is enabled (either Sigmoid or Thresholding)
    //   2) The input size is greater than the output size of SegNet
    bool runPostProcessing =
        m_pImpl->m_bi3dParams.sigmoidPostProcessing == ProcessingControl::ENABLE ||
        m_pImpl->m_bi3dParams.thresholdPostProcessing == ProcessingControl::ENABLE ||
        disparityConfidence.getDataSize() > m_pImpl->m_disparityConfidenceDevice.getDataSize();

    if (leftImage.getWidth() != m_pImpl->m_preprocessedLeftImageDevice.getWidth() ||
       leftImage.getHeight() != m_pImpl->m_preprocessedLeftImageDevice.getHeight()) {
        // Running the preprocessing

        m_pImpl->m_preprocess->execute(m_pImpl->m_preprocessedLeftImageDevice,
                                       m_pImpl->m_preprocessedRightImageDevice,
                                       leftImage, rightImage, stream);

        // Running feature extraction
        m_pImpl->m_featureInfer->execute(m_pImpl->m_leftFeaturesDevice,
                                         m_pImpl->m_preprocessedLeftImageDevice,
                                         stream);
        m_pImpl->m_featureInfer->execute(m_pImpl->m_rightFeaturesDevice,
                                         m_pImpl->m_preprocessedRightImageDevice,
                                         stream);
    } else {
        // Running feature extraction
        m_pImpl->m_featureInfer->execute(m_pImpl->m_leftFeaturesDevice,
                                         leftImage, stream);
        m_pImpl->m_featureInfer->execute(m_pImpl->m_rightFeaturesDevice,
                                         rightImage, stream);
    }

    if (m_pImpl->m_bi3dParams.edgeRefinement == ProcessingControl::ENABLE) {
        m_pImpl->m_hrInfer->execute(m_pImpl->m_hrFeaturesDevice,
                                    m_pImpl->m_preprocessedLeftImageDevice,
                                    stream);
    }

    // Moving the left image extracted features to a larger buffer for maximal
    // disparity calculation
    bi3d::Pad(m_pImpl->m_leftFeaturesPadDevice, m_pImpl->m_leftFeaturesDevice,
                0, 0, stream);

    for (std::size_t i = 0; i < disparityLevels.getSize(); ++i) {
        // Padding the right image extracted features to the designated disparity
        // level
        bi3d::Pad(m_pImpl->m_rightFeaturesPadDevice, m_pImpl->m_rightFeaturesDevice,
                    0, disparityLevels[i], stream);

        // Extracted features are passively concatenated into m_mergedFeaturesDevice
        // See: Bi3D::Bi3DImpl::Bi3DImpl()

        // Computing the depth confidence map on the given disparity level
        m_pImpl->m_segmentationInfer->execute(m_pImpl->m_confidenceMapDevice,
            m_pImpl->m_mergedFeaturesDevice,
            stream);

        // In the case where edge refinement is DISABLED and no postprocessing
        // is needed, the output buffer to Bi3D is directly used for computing
        // the disparity confidence. `m_pImpl->m_confidenceMapResizeDevice` is
        // a memory overlay of either the Bi3D output buffer `disparityConfidence`
        // or the swap buffer `m_pImpl->m_disparityConfidenceDevice` which reflects
        // the output format of SegNet. If no postprocessing is needed
        // `disparityConfidence` is directly used, else the swap buffer
        // `m_pImpl->m_disparityConfidenceDevice` is used to compute the
        // first pass dispairty confidence before post processing.
        // Cropping and resizing the disparity confidence to refinement size
        if (m_pImpl->m_bi3dParams.edgeRefinement == ProcessingControl::DISABLE) {
            if (!runPostProcessing) {
                m_pImpl->m_confidenceMapResizeDevice = {
                    disparityConfidence.getWidth(),
                    disparityConfidence.getHeight(),
                    disparityConfidence.getData()
                        + i*disparityConfidence.getStride(TensorDimension::CHANNEL),
                    disparityConfidence.isCPU()};
            } else {
                m_pImpl->m_confidenceMapResizeDevice = {
                    m_pImpl->m_disparityConfidenceDevice.getWidth(),
                    m_pImpl->m_disparityConfidenceDevice.getHeight(),
                    m_pImpl->m_disparityConfidenceDevice.getData()
                        + i*m_pImpl->m_disparityConfidenceDevice.getStride(
                            TensorDimension::CHANNEL),
                    m_pImpl->m_disparityConfidenceDevice.isCPU()};
            }
        }

        bi3d::CropAndResize(
            m_pImpl->m_confidenceMapResizeDevice,
            m_pImpl->m_confidenceMapDevice,
            {0, 0,
                static_cast<int>(m_pImpl->m_confidenceMapResizeDevice.getWidth()),
                static_cast<int>(m_pImpl->m_confidenceMapResizeDevice.getHeight())},
            {0, 0,
                static_cast<int>(m_pImpl->m_confidenceMapDevice.getWidth() -
                    m_pImpl->m_bi3dParams.maxDisparityLevels),
                static_cast<int>(m_pImpl->m_confidenceMapDevice.getHeight())},
            cvcore::tensor_ops::InterpolationType::INTERP_LINEAR, stream);

        // The confidence map and the HR features are passively concatenated into
        // m_fusedDisparityDevice
        // See: Bi3D::Bi3DImpl::Bi3DImpl()
        if (m_pImpl->m_bi3dParams.edgeRefinement == ProcessingControl::ENABLE) {
            // Creating an overlay of the memory sector to store the confidence results
            // channel-wise
            RefinedConfidence refinedConfidenceDevice{
                m_pImpl->m_disparityConfidenceDevice.getWidth(),
                m_pImpl->m_disparityConfidenceDevice.getHeight(),
                m_pImpl->m_disparityConfidenceDevice.getData()
                    + i*m_pImpl->m_disparityConfidenceDevice.getStride(TensorDimension::CHANNEL),
                m_pImpl->m_disparityConfidenceDevice.isCPU()};
            bi3d::Clear(refinedConfidenceDevice, stream);

            // Computing the refined confidence
            m_pImpl->m_refinementInfer->execute(refinedConfidenceDevice,
                m_pImpl->m_fusedDisparityDevice, stream);
        }
    }

    // Running the post-processing
    if (runPostProcessing) {
        m_pImpl->m_postprocess->execute(disparityConfidence,
            m_pImpl->m_disparityConfidenceDevice, stream);
    }
}

void Bi3D::execute(DisparityConfidence & disparityConfidence,
                   const InputImage & leftImage,
                   const InputImage & rightImage,
                   const DisparityLevels & disparityLevels,
                   cudaStream_t stream) {
    // Running the preprocessing
    m_pImpl->m_preprocess->execute(m_pImpl->m_preprocessedLeftImageDevice,
        m_pImpl->m_preprocessedRightImageDevice,
        leftImage,
        rightImage,
        stream);

    // Running processing
    this->execute(disparityConfidence,
        m_pImpl->m_preprocessedLeftImageDevice,
        m_pImpl->m_preprocessedRightImageDevice,
        disparityLevels, stream);
}

void Bi3D::execute(QuantizedDisparity & quantizedDisparity,
                   const InputImage & leftImage,
                   const InputImage & rightImage,
                   const DisparityLevels & disparityLevels,
                   cudaStream_t stream) {
    // Ensuring the buffers are appropreately allocated
    m_pImpl->resizeBuffers(leftImage.getWidth(), leftImage.getHeight());

    // Calling the floating point version of this method
    this->execute(m_pImpl->m_disparityConfidenceExchangeDevice,
        leftImage, rightImage, disparityLevels,
        stream);

    // Converting the confidence to U8
    bi3d::ConvertBitDepth(m_pImpl->m_quantizedDisparityDevice,
        m_pImpl->m_disparityConfidenceExchangeDevice,
        255.0, stream);

    // Copying the results to output
    QuantizedDisparity partial{quantizedDisparity.getWidth(),
        quantizedDisparity.getHeight(),
        quantizedDisparity.getChannelCount(),
        m_pImpl->m_quantizedDisparityDevice.getData(),
        m_pImpl->m_quantizedDisparityDevice.isCPU()};
    Copy(quantizedDisparity, partial, stream);
}

}  // namespace bi3d
}  // namespace isaac
}  // namespace nvidia
