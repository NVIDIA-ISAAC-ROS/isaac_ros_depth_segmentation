/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "extensions/bi3d/components/bi3d_postprocessor.cu.hpp"

namespace nvidia {
namespace isaac {
namespace bi3d {

__global__ void postprocessing_kernel(const float* input, float* output, int disparity,
                                      int imageHeight, int imageWidth) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t index = y * imageWidth + x;
  if (x < imageWidth && y < imageHeight) {
    if (input[index] == 1.0) {
      output[index] = input[index] * disparity;
    }
  }
}

uint16_t ceil_div(uint16_t numerator, uint16_t denominator) {
  uint32_t accumulator = numerator + denominator - 1;
  return accumulator / denominator;
}

void cuda_postprocess(const float* input, float* output, int disparity, int imageHeight,
                      int imageWidth) {
  dim3 block(16, 16);
  dim3 grid(ceil_div(imageWidth, 16), ceil_div(imageHeight, 16), 1);
  postprocessing_kernel<<<grid, block>>>(input, output, disparity, imageHeight, imageWidth);
}

}  // namespace bi3d
}  // namespace isaac
}  // namespace nvidia
