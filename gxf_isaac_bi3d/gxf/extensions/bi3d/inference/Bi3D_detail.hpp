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

#include <cuda_runtime.h>

#include "extensions/tensorops/core/Array.h"
#include "extensions/tensorops/core/BBox.h"
#include "extensions/tensorops/core/ImageUtils.h"
#include "extensions/tensorops/core/NppUtils.h"
#include "extensions/tensorops/core/Tensor.h"
#include "extensions/tensorops/core/Traits.h"

namespace nvidia {
namespace isaac {
namespace bi3d {

using TensorLayout = cvcore::tensor_ops::TensorLayout;
using ChannelType = cvcore::tensor_ops::ChannelType;
using ChannelCount =  cvcore::tensor_ops::ChannelCount;
using TensorDimension = cvcore::tensor_ops::TensorDimension;
using Tensor_CHW_C1_F32 = cvcore::tensor_ops::Tensor<TensorLayout::CHW, ChannelCount::C1,
    ChannelType::F32>;
using Tensor_HWC_C3_U8 = cvcore::tensor_ops::Tensor<TensorLayout::HWC, ChannelCount::C3,
    ChannelType::U8>;
using Tensor_CHW_CX_F32 = cvcore::tensor_ops::Tensor<TensorLayout::CHW, ChannelCount::CX,
    ChannelType::F32>;
using DisparityLevels = cvcore::tensor_ops::Array<int>;
using BBox = cvcore::tensor_ops::BBox;
using InputImage          = Tensor_HWC_C3_U8;
using PreprocessedImage   = cvcore::tensor_ops::to_planar_t<
    cvcore::tensor_ops::to_f32_t<InputImage>>;
using BatchPreprocessedImage = cvcore::tensor_ops::add_batch_t<PreprocessedImage>;
using FeatureResponse     = Tensor_CHW_CX_F32;
using BatchFeatureResponse = cvcore::tensor_ops::add_batch_t<FeatureResponse>;
using ConfidenceMap       = Tensor_CHW_C1_F32;
using GuidedConfidence    = cvcore::tensor_ops::to_cx_t<ConfidenceMap>;
using RefinedConfidence   = Tensor_CHW_C1_F32;
using DisparityConfidence = cvcore::tensor_ops::to_cx_t<RefinedConfidence>;
using QuantizedDisparity  = cvcore::tensor_ops::to_u8_t<DisparityConfidence>;
using NormalizeFunctor = cvcore::tensor_ops::ImageToNormalizedPlanarTensorOperator<
    TensorLayout::HWC, TensorLayout::CHW, ChannelCount::C3, ChannelType::U8>;

void _pad(Tensor_CHW_C1_F32& dst, const Tensor_CHW_C1_F32& src,
    int topBorderHeight, int leftBorderWidth, cudaStream_t stream);

template<ChannelCount CC>
void Pad(cvcore::tensor_ops::Tensor<TensorLayout::CHW, CC, ChannelType::F32> & dst,
    cvcore::tensor_ops::Tensor<TensorLayout::CHW, CC, ChannelType::F32> & src,
    const cvcore::tensor_ops::Array<int>& topBorderHeights,
    const cvcore::tensor_ops::Array<int>& leftBorderWidths,
    cudaStream_t stream) {
    using TensorType = Tensor_CHW_C1_F32;

    for (std::size_t c = 0; c < dst.getChannelCount(); ++c) {
        TensorType _dst{dst.getWidth(), dst.getHeight(),
                        dst.getData() + c*dst.getStride(TensorDimension::CHANNEL),
                        dst.isCPU()};
        TensorType _src{src.getWidth(), src.getHeight(),
                        src.getData() + c*src.getStride(TensorDimension::CHANNEL),
                        src.isCPU()};

        _pad(_dst, _src, topBorderHeights[c], leftBorderWidths[c], stream);
    }
}

template<ChannelCount CC>
void Pad(cvcore::tensor_ops::Tensor<TensorLayout::CHW, CC, ChannelType::F32> & dst,
    cvcore::tensor_ops::Tensor<TensorLayout::CHW, CC, ChannelType::F32> & src,
    const cvcore::tensor_ops::Array<int> & topBorderHeights, int leftBorderWidth,
    cudaStream_t stream) {
    using TensorType = Tensor_CHW_C1_F32;

    for (std::size_t c = 0; c < dst.getChannelCount(); ++c) {
        TensorType _dst{dst.getWidth(), dst.getHeight(),
                        dst.getData() + c*dst.getStride(TensorDimension::CHANNEL),
                        dst.isCPU()};
        TensorType _src{src.getWidth(), src.getHeight(),
                        src.getData() + c*src.getStride(TensorDimension::CHANNEL),
                        src.isCPU()};

        _pad(_dst, _src, topBorderHeights[c], leftBorderWidth, stream);
    }
}

template<ChannelCount CC>
void Pad(cvcore::tensor_ops::Tensor<TensorLayout::CHW, CC, ChannelType::F32> & dst,
    cvcore::tensor_ops::Tensor<TensorLayout::CHW, CC, ChannelType::F32> & src,
    int topBorderHeight, const cvcore::tensor_ops::Array<int>& leftBorderWidths,
    cudaStream_t stream) {
    using TensorType = Tensor_CHW_C1_F32;

    for (std::size_t c = 0; c < dst.getChannelCount(); ++c) {
        TensorType _dst{dst.getWidth(), dst.getHeight(),
                        dst.getData() + c*dst.getStride(TensorDimension::CHANNEL),
                        dst.isCPU()};
        TensorType _src{src.getWidth(), src.getHeight(),
                        src.getData() + c*src.getStride(TensorDimension::CHANNEL),
                        src.isCPU()};

        _pad(_dst, _src, topBorderHeight, leftBorderWidths[c], stream);
    }
}

template<ChannelCount CC>
void Pad(cvcore::tensor_ops::Tensor<TensorLayout::CHW, CC, ChannelType::F32> & dst,
    cvcore::tensor_ops::Tensor<TensorLayout::CHW, CC, ChannelType::F32> & src,
    int topBorderHeight, int leftBorderWidth, cudaStream_t stream) {
    using TensorType = Tensor_CHW_C1_F32;

    for (std::size_t c = 0; c < dst.getChannelCount(); ++c) {
        TensorType _dst{dst.getWidth(), dst.getHeight(),
                        dst.getData() + c*dst.getStride(TensorDimension::CHANNEL),
                        dst.isCPU()};
        TensorType _src{src.getWidth(), src.getHeight(),
                        src.getData() + c*src.getStride(TensorDimension::CHANNEL),
                        src.isCPU()};

        _pad(_dst, _src, topBorderHeight, leftBorderWidth, stream);
    }
}

template<ChannelCount CC>
void Pad(cvcore::tensor_ops::Tensor<TensorLayout::NCHW, CC, ChannelType::F32>& dst,
    cvcore::tensor_ops::Tensor<TensorLayout::NCHW, CC, ChannelType::F32>& src,
    int topBorderHeight, int leftBorderWidth, cudaStream_t stream) {
    using TensorType = Tensor_CHW_CX_F32;

    for (std::size_t n = 0; n < dst.getDepth(); ++n) {
        TensorType _dst{dst.getWidth(), dst.getHeight(), dst.getChannelCount(),
                        dst.getData() + n*dst.getStride(TensorDimension::DEPTH),
                        dst.isCPU()};
        TensorType _src{src.getWidth(), src.getHeight(), src.getChannelCount(),
                        src.getData() + n*src.getStride(TensorDimension::DEPTH),
                        src.isCPU()};

        Pad(_dst, _src, topBorderHeight, leftBorderWidth, stream);
    }
}

void _threshold(Tensor_CHW_C1_F32 & dst,
    const Tensor_CHW_C1_F32 & src,
    float valueLow, float valueHigh, float thresholdValue,
    cudaStream_t stream);

template<ChannelCount CC>
void Threshold(cvcore::tensor_ops::Tensor<TensorLayout::CHW, CC, ChannelType::F32> & dst,
    cvcore::tensor_ops::Tensor<TensorLayout::CHW, CC, ChannelType::F32> & src,
    float valueLow, float valueHigh, float thresholdValue,
    cudaStream_t stream) {
    using TensorType = Tensor_CHW_C1_F32;

    TensorType _dst{dst.getWidth(), dst.getHeight() * dst.getChannelCount(),
                    dst.getData(), dst.isCPU()};
    TensorType _src{src.getWidth(), src.getHeight() * src.getChannelCount(),
                    src.getData(), src.isCPU()};

    _threshold(_dst, _src, valueLow, valueHigh, thresholdValue, stream);
}

template<ChannelCount CC>
void Threshold(cvcore::tensor_ops::Tensor<TensorLayout::NCHW, CC, ChannelType::F32> & dst,
    cvcore::tensor_ops::Tensor<TensorLayout::NCHW, CC, ChannelType::F32> & src,
    float valueLow, float valueHigh, float thresholdValue,
    cudaStream_t stream) {
    using TensorType = Tensor_CHW_CX_F32;

    TensorType _dst{dst.getWidth(), dst.getHeight(),
                    dst.getChannelCount() * dst.getDepth(),
                    dst.getData(), dst.isCPU()};
    TensorType _src{src.getWidth(), src.getHeight(),
                    src.getChannelCount() * src.getDepth(),
                    src.getData(), src.isCPU()};

    Threshold(_dst, _src, valueLow, valueHigh, thresholdValue, stream);
}

template<ChannelCount CC>
void CropAndResize(cvcore::tensor_ops::Tensor<TensorLayout::CHW, CC, ChannelType::F32> & dst,
    cvcore::tensor_ops::Tensor<TensorLayout::CHW, CC, ChannelType::F32> & src,
    const BBox & dstROI, const BBox & srcROI,
    cvcore::tensor_ops::InterpolationType type, cudaStream_t stream) {
    using TensorType = Tensor_CHW_C1_F32;

    for (std::size_t c = 0; c < dst.getChannelCount(); ++c) {
        TensorType _dst{dst.getWidth(), dst.getHeight(),
                        dst.getData() + c*dst.getStride(TensorDimension::CHANNEL),
                        dst.isCPU()};
        TensorType _src{src.getWidth(), src.getHeight(),
                        src.getData() + c*src.getStride(TensorDimension::CHANNEL),
                        src.isCPU()};

        cvcore::tensor_ops::CropAndResize(_dst, _src, dstROI, srcROI, type, stream);
    }
}

template<ChannelCount CC>
void CropAndResize(cvcore::tensor_ops::Tensor<TensorLayout::NCHW, CC, ChannelType::F32> & dst,
    cvcore::tensor_ops::Tensor<TensorLayout::NCHW, CC, ChannelType::F32> & src,
    const BBox & dstROI, const BBox & srcROI,
    cvcore::tensor_ops::InterpolationType type, cudaStream_t stream) {
    using TensorType = Tensor_CHW_CX_F32;

    for (std::size_t n = 0; n < dst.getDepth(); ++n) {
        TensorType _dst{dst.getWidth(), dst.getHeight(), dst.getChannelCount(),
                        dst.getData() + n*dst.getStride(TensorDimension::DEPTH),
                        dst.isCPU()};
        TensorType _src{src.getWidth(), src.getHeight(), src.getChannelCount(),
                        src.getData() + n*src.getStride(TensorDimension::DEPTH),
                        src.isCPU()};

        CropAndResize(_dst, _src, dstROI, srcROI, type, stream);
    }
}

template<ChannelCount CC>
void Resize(cvcore::tensor_ops::Tensor<TensorLayout::CHW, CC, ChannelType::F32> & dst,
    cvcore::tensor_ops::Tensor<TensorLayout::CHW, CC, ChannelType::F32> & src,
    cvcore::tensor_ops::InterpolationType type, cudaStream_t stream) {
    using TensorType = Tensor_CHW_C1_F32;

    for (std::size_t c = 0; c < dst.getChannelCount(); ++c) {
        TensorType _dst{dst.getWidth(), dst.getHeight(),
                        dst.getData() + c*dst.getStride(TensorDimension::CHANNEL),
                        dst.isCPU()};
        TensorType _src{src.getWidth(), src.getHeight(),
                        src.getData() + c*src.getStride(TensorDimension::CHANNEL),
                        src.isCPU()};

        Resize(_dst, _src, false, type, stream);
    }
}

template<ChannelCount CC>
void Resize(cvcore::tensor_ops::Tensor<TensorLayout::NCHW, CC, ChannelType::F32> & dst,
    cvcore::tensor_ops::Tensor<TensorLayout::NCHW, CC, ChannelType::F32> & src,
    cvcore::tensor_ops::InterpolationType type, cudaStream_t stream) {
    using TensorType = Tensor_CHW_CX_F32;

    for (std::size_t n = 0; n < dst.getDepth(); ++n) {
        TensorType _dst{dst.getWidth(), dst.getHeight(), dst.getChannelCount(),
                        dst.getData() + n*dst.getStride(TensorDimension::DEPTH),
                        dst.isCPU()};
        TensorType _src{src.getWidth(), src.getHeight(), src.getChannelCount(),
                        src.getData() + n*src.getStride(TensorDimension::DEPTH),
                        src.isCPU()};

        Resize(_dst, _src, type, stream);
    }
}

void _clear(Tensor_CHW_C1_F32& dst, cudaStream_t stream);

template<ChannelCount CC>
void Clear(cvcore::tensor_ops::Tensor<TensorLayout::CHW, CC, ChannelType::F32> & dst,
    cudaStream_t stream) {
    using TensorType = Tensor_CHW_C1_F32;

    for (std::size_t c = 0; c < dst.getChannelCount(); ++c) {
        TensorType _dst{dst.getWidth(), dst.getHeight(),
                        dst.getData() + c*dst.getStride(TensorDimension::CHANNEL),
                        dst.isCPU()};

        _clear(_dst, stream);
    }
}

template<ChannelCount CC>
void ConvertBitDepth(cvcore::tensor_ops::Tensor<TensorLayout::CHW, CC, ChannelType::U8> & dst,
    cvcore::tensor_ops::Tensor<TensorLayout::CHW, CC, ChannelType::F32> & src,
    float scale, cudaStream_t stream) {
    using TensorSrcType = Tensor_CHW_C1_F32;
    using TensorDstType = cvcore::tensor_ops::to_u8_t<TensorSrcType>;

    for (std::size_t c = 0; c < dst.getChannelCount(); ++c) {
        TensorDstType _dst{dst.getWidth(), dst.getHeight(),
                           dst.getData() + c*dst.getStride(TensorDimension::CHANNEL),
                           dst.isCPU()};
        TensorSrcType _src{src.getWidth(), src.getHeight(),
                           src.getData() + c*src.getStride(TensorDimension::CHANNEL),
                           src.isCPU()};

        ConvertBitDepth(_dst, _src, scale, stream);
    }
}

void _sigmoid(Tensor_CHW_C1_F32& dst,
    Tensor_CHW_C1_F32& src, cudaStream_t stream);

void _sigmoid(Tensor_CHW_C1_F32& dst, Tensor_CHW_C1_F32& src);

template<ChannelCount CC>
void Sigmoid(cvcore::tensor_ops::Tensor<TensorLayout::CHW, CC, ChannelType::F32> & dst,
    cvcore::tensor_ops::Tensor<TensorLayout::CHW, CC, ChannelType::F32> & src,
             cudaStream_t stream) {
    using TensorType = Tensor_CHW_C1_F32;

    for (std::size_t c = 0; c < dst.getChannelCount(); ++c) {
        TensorType _dst{dst.getWidth(), dst.getHeight(),
                        dst.getData() + c*dst.getStride(TensorDimension::CHANNEL),
                        dst.isCPU()};
        TensorType _src{src.getWidth(), src.getHeight(),
                        src.getData() + c*src.getStride(TensorDimension::CHANNEL),
                        src.isCPU()};

        _sigmoid(_dst, _src, stream);
    }
}

template<ChannelCount CC>
void Sigmoid(cvcore::tensor_ops::Tensor<TensorLayout::CHW, CC, ChannelType::F32>& dst,
    cvcore::tensor_ops::Tensor<TensorLayout::CHW, CC, ChannelType::F32>& src) {

    for (std::size_t c = 0; c < dst.getChannelCount(); ++c) {
        Tensor_CHW_C1_F32 _dst{dst.getWidth(), dst.getHeight(),
                        dst.getData() + c*dst.getStride(TensorDimension::CHANNEL),
                        dst.isCPU()};
        Tensor_CHW_C1_F32 _src{src.getWidth(), src.getHeight(),
                        src.getData() + c*src.getStride(TensorDimension::CHANNEL),
                        src.isCPU()};

        _sigmoid(_dst, _src);
    }
}

}  // namespace bi3d
}  // namespace isaac
}  // namespace nvidia
