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
#include "bi3d_postprocessor.cu.hpp"

namespace nvidia
{
namespace isaac_ros
{

__global__ void postprocessing_kernel(
  const float * input, float * output, int disparity,
  int imageHeight, int imageWidth)
{
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t index = y * imageWidth + x;
  if (x < imageWidth && y < imageHeight) {
    if (input[index] == 1.0) {
      output[index] = input[index] * disparity;
    }
  }
}

uint16_t ceil_div(uint16_t numerator, uint16_t denominator)
{
  uint32_t accumulator = numerator + denominator - 1;
  return accumulator / denominator;
}

void cuda_postprocess(
  const float * input, float * output, int disparity, int imageHeight,
  int imageWidth)
{
  dim3 block(16, 16);
  dim3 grid(ceil_div(imageWidth, 16), ceil_div(imageHeight, 16), 1);
  postprocessing_kernel << < grid, block >> > (input, output, disparity, imageHeight, imageWidth);
}

}  // namespace isaac_ros
}  // namespace nvidia
