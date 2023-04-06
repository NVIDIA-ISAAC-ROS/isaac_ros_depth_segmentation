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

#ifndef CVCORE_BI3D_DETAIL_H_
#define CVCORE_BI3D_DETAIL_H_

#include <cuda_runtime.h>

#include <cv/core/Array.h>
#include <cv/core/Tensor.h>
#include <cv/core/Traits.h>

#include <cv/tensor_ops/ImageUtils.h>
#include <cv/tensor_ops/NppUtils.h>

namespace cvcore { namespace bi3d { namespace detail {

using DisparityLevels     = Array<int>;
using InputImage          = Tensor<HWC, C3, U8>;
using PreprocessedImage   = traits::to_planar_t<traits::to_f32_t<InputImage>>;
using BatchPreprocessedImage = traits::add_batch_t<PreprocessedImage>;
using FeatureResponse     = Tensor<CHW, CX, F32>;
using BatchFeatureResponse = traits::add_batch_t<FeatureResponse>;
using ConfidenceMap       = Tensor<CHW, C1, F32>;
using GuidedConfidence    = traits::to_cx_t<ConfidenceMap>;
using RefinedConfidence   = Tensor<CHW, C1, F32>;
using DisparityConfidence = traits::to_cx_t<RefinedConfidence>;
using QuantizedDisparity  = traits::to_u8_t<DisparityConfidence>;

using NormalizeFunctor = tensor_ops::ImageToNormalizedPlanarTensorOperator<HWC, CHW, C3, U8>;

void _pad(Tensor<CHW, C1, F32> & dst, const Tensor<CHW, C1, F32> & src,
          int topBorderHeight, int leftBorderWidth, cudaStream_t stream);

template<ChannelCount CC>
void Pad(Tensor<CHW, CC, F32> & dst, Tensor<CHW, CC, F32> & src,
         const Array<int> & topBorderHeights, const Array<int> & leftBorderWidths,
         cudaStream_t stream)
{
    using TensorType = Tensor<CHW, C1, F32>;

    for(std::size_t c = 0; c < dst.getChannelCount(); ++c)
    {
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
void Pad(Tensor<CHW, CC, F32> & dst, Tensor<CHW, CC, F32> & src,
         const Array<int> & topBorderHeights, int leftBorderWidth, cudaStream_t stream)
{
    using TensorType = Tensor<CHW, C1, F32>;

    for(std::size_t c = 0; c < dst.getChannelCount(); ++c)
    {
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
void Pad(Tensor<CHW, CC, F32> & dst, Tensor<CHW, CC, F32> & src,
         int topBorderHeight, const Array<int> & leftBorderWidths, cudaStream_t stream)
{
    using TensorType = Tensor<CHW, C1, F32>;

    for(std::size_t c = 0; c < dst.getChannelCount(); ++c)
    {
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
void Pad(Tensor<CHW, CC, F32> & dst, Tensor<CHW, CC, F32> & src,
         int topBorderHeight, int leftBorderWidth, cudaStream_t stream)
{
    using TensorType = Tensor<CHW, C1, F32>;

    for(std::size_t c = 0; c < dst.getChannelCount(); ++c)
    {
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
void Pad(Tensor<NCHW, CC, F32> & dst, Tensor<NCHW, CC, F32> & src,
         int topBorderHeight, int leftBorderWidth, cudaStream_t stream)
{
    using TensorType = Tensor<CHW, CX, F32>;

    for(std::size_t n = 0; n < dst.getDepth(); ++n)
    {
        TensorType _dst{dst.getWidth(), dst.getHeight(), dst.getChannelCount(),
                        dst.getData() + n*dst.getStride(TensorDimension::DEPTH),
                        dst.isCPU()};
        TensorType _src{src.getWidth(), src.getHeight(), src.getChannelCount(),
                        src.getData() + n*src.getStride(TensorDimension::DEPTH),
                        src.isCPU()};
        
        Pad(_dst, _src, topBorderHeight, leftBorderWidth, stream);
    }
}

void _threshold(Tensor<CHW, C1, F32> & dst, const Tensor<CHW, C1, F32> & src,
                float valueLow, float valueHigh, float thresholdValue,
                cudaStream_t stream);

template<ChannelCount CC>
void Threshold(Tensor<CHW, CC, F32> & dst, Tensor<CHW, CC, F32> & src,
               float valueLow, float valueHigh, float thresholdValue,
               cudaStream_t stream)
{
    using TensorType = Tensor<CHW, C1, F32>;

    TensorType _dst{dst.getWidth(), dst.getHeight() * dst.getChannelCount(),
                    dst.getData(), dst.isCPU()};
    TensorType _src{src.getWidth(), src.getHeight() * src.getChannelCount(),
                    src.getData(), src.isCPU()};
    
    _threshold(_dst, _src, valueLow, valueHigh, thresholdValue, stream);
}

template<ChannelCount CC>
void Threshold(Tensor<NCHW, CC, F32> & dst, Tensor<NCHW, CC, F32> & src,
               float valueLow, float valueHigh, float thresholdValue,
               cudaStream_t stream)
{
    using TensorType = Tensor<CHW, CX, F32>;
    
    TensorType _dst{dst.getWidth(), dst.getHeight(),
                    dst.getChannelCount() * dst.getDepth(),
                    dst.getData(), dst.isCPU()};
    TensorType _src{src.getWidth(), src.getHeight(),
                    src.getChannelCount() * src.getDepth(),
                    src.getData(), src.isCPU()};
    
    Threshold(_dst, _src, valueLow, valueHigh, thresholdValue, stream);
}

template<ChannelCount CC>
void CropAndResize(Tensor<CHW, CC, F32> & dst, Tensor<CHW, CC, F32> & src,
                   const BBox & dstROI, const BBox & srcROI,
                   tensor_ops::InterpolationType type, cudaStream_t stream)
{
    using TensorType = Tensor<CHW, C1, F32>;

    for(std::size_t c = 0; c < dst.getChannelCount(); ++c)
    {
        TensorType _dst{dst.getWidth(), dst.getHeight(),
                        dst.getData() + c*dst.getStride(TensorDimension::CHANNEL),
                        dst.isCPU()};
        TensorType _src{src.getWidth(), src.getHeight(),
                        src.getData() + c*src.getStride(TensorDimension::CHANNEL),
                        src.isCPU()};
        
        tensor_ops::CropAndResize(_dst, _src, dstROI, srcROI, type, stream);
    }
}

template<ChannelCount CC>
void CropAndResize(Tensor<NCHW, CC, F32> & dst, Tensor<NCHW, CC, F32> & src,
                   const BBox & dstROI, const BBox & srcROI,
                   tensor_ops::InterpolationType type, cudaStream_t stream)
{
    using TensorType = Tensor<CHW, CX, F32>;

    for(std::size_t n = 0; n < dst.getDepth(); ++n)
    {
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
void Resize(Tensor<CHW, CC, F32> & dst, Tensor<CHW, CC, F32> & src,
            tensor_ops::InterpolationType type, cudaStream_t stream)
{
    using TensorType = Tensor<CHW, C1, F32>;

    for(std::size_t c = 0; c < dst.getChannelCount(); ++c)
    {
        TensorType _dst{dst.getWidth(), dst.getHeight(),
                        dst.getData() + c*dst.getStride(TensorDimension::CHANNEL),
                        dst.isCPU()};
        TensorType _src{src.getWidth(), src.getHeight(),
                        src.getData() + c*src.getStride(TensorDimension::CHANNEL),
                        src.isCPU()};
        
        tensor_ops::Resize(_dst, _src, false, type, stream);
    }
}

template<ChannelCount CC>
void Resize(Tensor<NCHW, CC, F32> & dst, Tensor<NCHW, CC, F32> & src,
            tensor_ops::InterpolationType type, cudaStream_t stream)
{
    using TensorType = Tensor<CHW, CX, F32>;

    for(std::size_t n = 0; n < dst.getDepth(); ++n)
    {
        TensorType _dst{dst.getWidth(), dst.getHeight(), dst.getChannelCount(),
                        dst.getData() + n*dst.getStride(TensorDimension::DEPTH),
                        dst.isCPU()};
        TensorType _src{src.getWidth(), src.getHeight(), src.getChannelCount(),
                        src.getData() + n*src.getStride(TensorDimension::DEPTH),
                        src.isCPU()};
        
        Resize(_dst, _src, type, stream);
    }
}

void _clear(Tensor<CHW, C1, F32> & dst, cudaStream_t stream);

template<ChannelCount CC>
void Clear(Tensor<CHW, CC, F32> & dst, cudaStream_t stream)
{
    using TensorType = Tensor<CHW, C1, F32>;

    for(std::size_t c = 0; c < dst.getChannelCount(); ++c)
    {
        TensorType _dst{dst.getWidth(), dst.getHeight(),
                        dst.getData() + c*dst.getStride(TensorDimension::CHANNEL),
                        dst.isCPU()};
        
        _clear(_dst, stream);
    }
}

template<ChannelCount CC>
void ConvertBitDepth(Tensor<CHW, CC, U8> & dst, Tensor<CHW, CC, F32> & src,
                     float scale, cudaStream_t stream)
{
    using TensorSrcType = Tensor<HWC, C1, F32>;
    using TensorDstType = traits::to_u8_t<TensorSrcType>;

    for(std::size_t c = 0; c < dst.getChannelCount(); ++c)
    {
        TensorDstType _dst{dst.getWidth(), dst.getHeight(),
                           dst.getData() + c*dst.getStride(TensorDimension::CHANNEL),
                           dst.isCPU()};
        TensorSrcType _src{src.getWidth(), src.getHeight(),
                           src.getData() + c*src.getStride(TensorDimension::CHANNEL),
                           src.isCPU()};
        
        tensor_ops::ConvertBitDepth(_dst, _src, scale, stream);
    }
}

void _sigmoid(Tensor<CHW, C1, F32> & dst, Tensor<CHW, C1, F32> & src,
              cudaStream_t stream);

void _sigmoid(Tensor<CHW, C1, F32> & dst, Tensor<CHW, C1, F32> & src);

template<ChannelCount CC>
void Sigmoid(Tensor<CHW, CC, F32> & dst, Tensor<CHW, CC, F32> & src,
             cudaStream_t stream)
{
    using TensorType = Tensor<CHW, C1, F32>;

    for(std::size_t c = 0; c < dst.getChannelCount(); ++c)
    {
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
void Sigmoid(Tensor<CHW, CC, F32> & dst, Tensor<CHW, CC, F32> & src)
{
    using TensorType = Tensor<CHW, C1, F32>;

    for(std::size_t c = 0; c < dst.getChannelCount(); ++c)
    {
        TensorType _dst{dst.getWidth(), dst.getHeight(),
                        dst.getData() + c*dst.getStride(TensorDimension::CHANNEL),
                        dst.isCPU()};
        TensorType _src{src.getWidth(), src.getHeight(),
                        src.getData() + c*src.getStride(TensorDimension::CHANNEL),
                        src.isCPU()};
        
        _sigmoid(_dst, _src);
    }
}

}}} // namespace cvcore::bi3d::detail

#endif // CVCORE_BI3D_DETAIL_H_
