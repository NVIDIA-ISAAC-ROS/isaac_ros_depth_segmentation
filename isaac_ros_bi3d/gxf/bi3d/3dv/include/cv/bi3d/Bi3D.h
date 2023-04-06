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

#ifndef CVCORE_BI3D_H_
#define CVCORE_BI3D_H_

#include <memory>

#include <cuda_runtime.h>

#include <cv/core/Array.h>
#include <cv/core/Tensor.h>
#include <cv/core/Model.h>
#include <cv/core/ComputeEngine.h>

#include <cv/inferencer/Inferencer.h>

namespace cvcore { namespace bi3d {

enum class ProcessingControl : std::uint8_t
{
    DISABLE = 0,
    ENABLE = 1
};

/**
 * Interface for Loading and running inference on bi3d network.
 */
class Bi3D
{
    public:
        using InferencerParams = inferencer::TensorRTInferenceParams;

        /**
         * Default Image Processing Params for Bi3D.
         */
        static const ImagePreProcessingParams defaultPreProcessorParams;

        /**
         * Default Model Input Params for Bi3D.
         */
        static const ModelInputParams defaultModelInputParams;

        /**
         * Default inference Params for Bi3D.
         */
        static const InferencerParams defaultFeatureParams;
        static const InferencerParams defaultHRParams;
        static const InferencerParams defaultRefinementParams;
        static const InferencerParams defaultSegmentationParams;
        
        /**
         * Bi3D extra params
         */
        struct Bi3DParams
        {
            std::size_t       maxDisparityLevels;       // Maximum pixel-wise shift between left and right images
            ProcessingControl edgeRefinement;           // Switch to turn on/off edge refinement models (FeatNetHR and
                                                        // RefineNet) in inferencing
            ProcessingControl sigmoidPostProcessing;    // Switch to turn on/off sigmoid operation in postprocessing
            ProcessingControl thresholdPostProcessing;  // Switch to turn on/off thresholding in postprocessing
            float thresholdValueLow;                    // The low value set by threshold postprocessing
            float thresholdValueHigh;                   // The high value set by threshold postprocessing
            float threshold;                            // The threshold value that casts pixel values to either low or high
        };
        static const Bi3DParams defaultBi3DParams;

        /**
         * Defaule Constructor of Bi3D.
         */
        Bi3D() = delete;

        /**
         * Constructor of Bi3D.
         * @param params custom params for the network.
         */
        Bi3D(const ImagePreProcessingParams & preProcessorParams,
             const ModelInputParams & modelInputParams,
             const InferencerParams & featureParams,
             const InferencerParams & hrParams,
             const InferencerParams & refinementParams,
             const InferencerParams & segmentationParams,
             const Bi3DParams & bi3dParams,
             cudaStream_t stream = 0);

        /**
         * Destructor of Bi3D.
         */
        ~Bi3D();

        /**
         * Main interface to run inference.
         * @param quantizedDisparity depth segmentation for input left and right images
         * @param leftImage left image tensor in RGB HWC format.
         * @param rightImage right image tensor in RGB HWC format.
         * @param disparityLevels diparity value to run depth estimation based on.
         */
        void execute(Tensor<CHW, CX, U8> & quantizedDisparity,
                     const Tensor<HWC, C3, U8> & leftImage,
                     const Tensor<HWC, C3, U8> & rightImage,
                     const Array<int> & disparityLevels,
                     cudaStream_t stream = 0);

        /**
         * Main interface to run inference.
         * @param disparityConfidence depth segmentation for input left and right images
         * @param leftImage left image tensor in RGB HWC format.
         * @param rightImage right image tensor in RGB HWC format.
         * @param disparityLevels diparity value to run depth estimation based on.
         */
        void execute(Tensor<CHW, CX, F32> & disparityConfidence,
                     const Tensor<HWC, C3, U8> & leftImage,
                     const Tensor<HWC, C3, U8> & rightImage,
                     const Array<int> & disparityLevels,
                     cudaStream_t stream = 0);

        /**
         * Main interface to run inference.
         * @param disparityConfidence depth segmentation for input left and right images
         * @param leftImage left image tensor in RGB CHW format.
         * @param rightImage right image tensor in RGB CHW format.
         * @param disparityLevels diparity value to run depth estimation based on.
         */
        void execute(Tensor<CHW, CX, F32> & disparityConfidence,
                     const Tensor<CHW, C3, F32> & leftImage,
                     const Tensor<CHW, C3, F32> & rightImage,
                     const Array<int> & disparityLevels,
                     cudaStream_t stream = 0);

    private:
        struct Bi3DImpl;

        std::unique_ptr<Bi3DImpl> m_pImpl;
};

/**
 * Interface for running pre-processing for Bi3D.
 */
class Bi3DPreProcessor 
{
    public:
        /**
         * Removing the default constructor for Bi3DPreProcessor.
         */
        Bi3DPreProcessor() = delete;

        /**
         * Constructor for Bi3DPreProcessor.
         * @param preProcessorParams Image preprocessing parameters.
         * @param modelInputParams Model input parameters.
         */
        Bi3DPreProcessor(const ImagePreProcessingParams & preProcessorParams, 
                         const ModelInputParams & modelInputParams,
                         cudaStream_t stream = 0);

        /**
         * Destructor for Bi3DPreProcessor.
         */
        ~Bi3DPreProcessor();

        void execute(Tensor<CHW, C3, F32> & preprocessedLeftImage,
                     Tensor<CHW, C3, F32> & preprocessedRightImage,
                     const Tensor<HWC, C3, U8> & leftImage,
                     const Tensor<HWC, C3, U8> & rightImage,
                     cudaStream_t stream = 0);

        void execute(Tensor<CHW, C3, F32> & preprocessedLeftImage,
                     Tensor<CHW, C3, F32> & preprocessedRightImage,
                     const Tensor<CHW, C3, F32> & leftImage,
                     const Tensor<CHW, C3, F32> & rightImage,
                     cudaStream_t stream = 0);

    private:
        struct Bi3DPreProcessorImpl;

        std::unique_ptr<Bi3DPreProcessorImpl> m_pImpl;
};

/**
 * Interface for running post-processing for Bi3D.
 */
class Bi3DPostProcessor 
{
    public:
        /**
         * Removing the default constructor for Bi3DPostProcessor.
         */
        Bi3DPostProcessor() = delete;

        /**
         * Constructor for Bi3DPostProcessor.
         * @param modelInputParams Model input parameters.
         * @param bi3dParams Model parameters unique to this model.
         */
        Bi3DPostProcessor(const ModelInputParams & modelInputParams,
                          const Bi3D::Bi3DParams & bi3dParams,
                          cudaStream_t stream = 0);

        /**
         * Destructor for Bi3DPostProcessor.
         */
        ~Bi3DPostProcessor();

        void execute(Tensor<CHW, CX, F32> & disparityConfidence,
                     Tensor<CHW, CX, F32> & rawDisparityConfidence,
                     cudaStream_t stream = 0);

    private:
        struct Bi3DPostProcessorImpl;

        std::unique_ptr<Bi3DPostProcessorImpl> m_pImpl;
};

}} // namespace cvcore::bi3d

#endif // CVCORE_BI3D_H_
