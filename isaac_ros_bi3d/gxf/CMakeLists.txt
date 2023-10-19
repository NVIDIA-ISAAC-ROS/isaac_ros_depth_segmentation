# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

project(gxf_cvcore_bi3d LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fPIC -w)
endif()

enable_language(CUDA)

# Dependencies
find_package(CUDAToolkit)
find_package(yaml-cpp)
find_package(GXF ${ISAAC_ROS_GXF_VERSION} MODULE REQUIRED
    COMPONENTS
    core
    cuda
    multimedia
    serialization
)
find_package(TENSORRT)
find_package(Eigen3)
find_package(isaac_ros_image_proc REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR TRUE)


# Create extension
add_library(gxf_cvcore_bi3d SHARED
  extensions/bi3d/bi3d.cpp
  extensions/bi3d/components/bi3d_postprocessor.cpp
  extensions/bi3d/components/bi3d_inference.cpp
  extensions/bi3d/components/bi3d_postprocessor.cu
  extensions/bi3d/components/bi3d_depthview.cpp

  extensions/bi3d/inference/Bi3D_detail.cpp
  extensions/bi3d/inference/Bi3D_detail.hpp
  extensions/bi3d/inference/Bi3D.cpp
  extensions/bi3d/inference/Bi3D.h
  extensions/bi3d/inference/Bi3DPreProcessor.cpp
  extensions/bi3d/inference/Bi3DPostProcessor.cpp
)
target_include_directories(gxf_cvcore_bi3d PRIVATE ${isaac_ros_image_proc_INCLUDE_DIRS})

add_library(corelib STATIC
  # Inferencer (Bi3D only)
  gems/dnn_inferencer/inferencer/TensorRTInferencer.cpp
  gems/dnn_inferencer/inferencer/TensorRTUtils.cpp
  gems/dnn_inferencer/inferencer/Inferencer.cpp
  gems/dnn_inferencer/inferencer/Errors.cpp
)
target_include_directories(corelib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${isaac_ros_image_proc_INCLUDE_DIRS})

target_link_libraries(corelib PUBLIC
    Eigen3::Eigen
    GXF::core
    GXF::cuda
    GXF::isaac_messages
    GXF::multimedia
    CUDA::cudart
    CUDA::nppc
    CUDA::nppial
    CUDA::nppicc
    CUDA::nppidei
    CUDA::nppif
    CUDA::nppig
    CUDA::nppisu
    CUDA::nppitc
    TENSORRT::nvinfer
    yaml-cpp
    
)

target_compile_options(gxf_cvcore_bi3d PUBLIC -fPIC)

target_link_libraries(gxf_cvcore_bi3d
    corelib
    isaac_ros_image_proc::gxf_tensorops
)