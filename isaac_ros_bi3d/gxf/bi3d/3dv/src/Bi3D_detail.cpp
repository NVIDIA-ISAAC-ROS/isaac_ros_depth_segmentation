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

#include "Bi3D_detail.h"

#include <limits>
#include <cmath>

#include <nppi_data_exchange_and_initialization.h>
#include <nppi_arithmetic_and_logical_operations.h>
#include <nppi_threshold_and_compare_operations.h>

namespace cvcore { namespace bi3d { namespace detail {

void _pad(Tensor<CHW, C1, F32> & dst, const Tensor<CHW, C1, F32> & src,
          int topBorderHeight, int leftBorderWidth, cudaStream_t stream)
{
    // Using NPP function: ``nppiCopyConstBorder_32f_C1R_Ctx''
    // Spec ==> NppStatus nppiCopyConstBorder_32f_C1R_Ctx(const Npp32f * pSrc,
    //                                                    int nSrcStep,
    //                                                    NppiSize oSrcSizeROI,
    //                                                    Npp32f * pDst,
    //                                                    int nDstStep,
    //                                                    NppiSize oDstSizeROI,
    //                                                    int nTopBorderHeight,
    //                                                    int nLeftBorderWidth,
    //                                                    Npp32f nValue,
    //                                                    NppStreamContext nppStreamCtx)
    //
    // Parameters
    //      pSrc                Source-Image Pointer.
    //      nSrcStep            Source-Image Stride in bytes.
    //      oSrcSizeROI         Size (width, height) of the source region in pixels.
    //      pDst                Destination-Image Pointer.
    //      nDstStep            Destination-Image Stride in bytes.
    //      oDstSizeROI         Size (width, height) of the destination region, i.e. the region that gets
    //                          filled with data from the source image (inner part) and constant border 
    //                          color (outer part).
    //      nTopBorderHeight    Height (in pixels) of the top border. The number of pixel rows at the top 
    //                          of the destination ROI that will be filled with the constant border
    //                          color. nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - 
    //                          oSrcSizeROI.height.
    //      nLeftBorderWidth    Width (in pixels) of the left border. The width of the border at the 
    //                          right side of the destination ROI is implicitly defined by the size of 
    //                          the source ROI: nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth 
    //                          - oSrcSizeROI.width.
    //      nValue              The pixel value to be set for border pixels for single channel functions.
    //      nppStreamCtx        NPP Application Managed Stream Context.
    nppiCopyConstBorder_32f_C1R_Ctx(
        static_cast<const float*>(src.getData()),
        src.getStride(TensorDimension::HEIGHT) * sizeof(float),
        {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())},
        static_cast<float*>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(float),
        {static_cast<int>(dst.getWidth()), static_cast<int>(dst.getHeight())},
        topBorderHeight, leftBorderWidth, 0.0, tensor_ops::GetNppStreamContext(stream));
}

void _threshold(Tensor<CHW, C1, F32> & dst, const Tensor<CHW, C1, F32> & src,
                float valueLow, float valueHigh, float thresholdValue,
                cudaStream_t stream)
{
    // Using NPP function: ``nppiThreshold_LTValGTVal_32f_C1R_Ctx''
    // Spec ==> NppStatus nppiThreshold_LTValGTVal_32f_C1R_Ctx(const Npp32f * pSrc,
    //                                                         int nSrcStep,
    //                                                         Npp32f * pDst,
    //                                                         int nDstStep,
    //                                                         NppiSize oSizeROI,
    //                                                         Npp32f nThresholdLT,
    //                                                         Npp32f nValueLT,
    //                                                         Npp32f nThresholdGT,
    //                                                         Npp32f nValueGT,
    //                                                         NppStreamContext nppStreamCtx)
    //
    // Parameters
    //      pSrc                Source-Image Pointer.
    //      nSrcStep            Source-Image Stride in bytes.
    //      pDst                Destination-Image Pointer.
    //      nDstStep            Destination-Image Stride in bytes.
    //      oSizeROI            Size (width, height) of the destination region, i.e. the region that gets
    //                          filled with data from the source image
    //      nThresholdLT        The thresholdLT value.
    //      nValueLT            The thresholdLT replacement value.
    //      nThresholdGT        The thresholdGT value.
    //      nValueGT            The thresholdGT replacement value.
    //      nppStreamCtx        NPP Application Managed Stream Context.
    nppiThreshold_LTValGTVal_32f_C1R_Ctx(
        static_cast<const float*>(src.getData()),
        src.getStride(TensorDimension::HEIGHT) * sizeof(float),
        static_cast<float*>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(float),
        {static_cast<int>(dst.getWidth()), static_cast<int>(dst.getHeight())},
        thresholdValue * (1.0 + std::numeric_limits<Npp32f>::epsilon()), valueLow,
        thresholdValue, valueHigh,
        tensor_ops::GetNppStreamContext(stream));
}

void _clear(Tensor<CHW, C1, F32> & dst, cudaStream_t stream)
{
    // Using NPP function: ``nppiSet_32f_C1R_Ctx''
    // Spec ==> NppStatus nppiSet_32f_C1R_Ctx(const Npp32f nValue
    //                                        Npp32f * pDst,
    //                                        int nDstStep,
    //                                        NppiSize oSizeROI,
    //                                        NppStreamContext nppStreamCtx)
    //
    // Parameters
    //      nValue          The pixel value to be set.
    //      pDst            Destination-Image Pointer.
    //      nDstStep        Destination-Image Stride in bytes.
    //      oSizeROI        Size (width, height) of the destination region, i.e. the region that gets
    //                          filled with nValue.
    //      nppStreamCtx    NPP Application Managed Stream Context.
    nppiSet_32f_C1R_Ctx(
        0.0,
        static_cast<float*>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(float),
        {static_cast<int>(dst.getWidth()), static_cast<int>(dst.getHeight())},
        tensor_ops::GetNppStreamContext(stream));
}

void _sigmoid(Tensor<CHW, C1, F32> & dst, Tensor<CHW, C1, F32> & src,
              cudaStream_t stream)
{
    auto context = tensor_ops::GetNppStreamContext(stream);

    // This is a bit of a complex series of operations, here's an exact model
    //
    // Explicit:
    //   DST = exp(-1 * ln(1 + exp(-1 * SRC)))
    // 
    // Properties:
    //   1) exp(ln(x)) = x
    //   2) ln(x^y) = y * ln(x)
    //   3) x^(-1) = 1 / x
    //
    // Expected:
    //   DST = 1 / (1 + exp(-1 * SRC))
    nppiMulC_32f_C1R_Ctx(
        static_cast<const float*>(src.getData()),
        src.getStride(TensorDimension::HEIGHT) * sizeof(float),
        -1.0,
        static_cast<float*>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(float),
        {static_cast<int>(dst.getWidth()), static_cast<int>(dst.getHeight())},
        context);
    nppiExp_32f_C1IR_Ctx(
        static_cast<float*>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(float),
        {static_cast<int>(dst.getWidth()), static_cast<int>(dst.getHeight())},
        context);
    nppiAddC_32f_C1IR_Ctx(
        1.0,
        static_cast<float*>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(float),
        {static_cast<int>(dst.getWidth()), static_cast<int>(dst.getHeight())},
        context);
    nppiLn_32f_C1IR_Ctx(
        static_cast<float*>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(float),
        {static_cast<int>(dst.getWidth()), static_cast<int>(dst.getHeight())},
        context);
    nppiMulC_32f_C1IR_Ctx(
        -1.0,
        static_cast<float*>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(float),
        {static_cast<int>(dst.getWidth()), static_cast<int>(dst.getHeight())},
        context);
    nppiExp_32f_C1IR_Ctx(
        static_cast<float*>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(float),
        {static_cast<int>(dst.getWidth()), static_cast<int>(dst.getHeight())},
        context);
}

void _sigmoid(Tensor<CHW, C1, F32> & dst, Tensor<CHW, C1, F32> & src)
{
    std::transform(src.getData(), src.getData() + src.getDataSize() / sizeof(float),
                   dst.getData(), [](float value) -> float {return 1.0 / (1.0 + std::exp(-1.0 * value));});
}

}}} // namespace cvcore::bi3d::detail
