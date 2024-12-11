# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.22.1)
project(gxf_isaac_bi3d LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fPIC -w)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

enable_language(CUDA)

# Dependencies
find_package(CUDAToolkit)
find_package(yaml-cpp)
find_package(TENSORRT)
find_package(Eigen3)

# Bi3D extension
ament_auto_add_library(${PROJECT_NAME} SHARED
  gxf/extensions/bi3d/bi3d.cpp
  gxf/extensions/bi3d/components/bi3d_postprocessor.cpp
  gxf/extensions/bi3d/components/bi3d_inference.cpp
  gxf/extensions/bi3d/components/bi3d_postprocessor.cu
  gxf/extensions/bi3d/components/bi3d_depthview.cpp

  gxf/extensions/bi3d/inference/Bi3D_detail.cpp
  gxf/extensions/bi3d/inference/Bi3D_detail.hpp
  gxf/extensions/bi3d/inference/Bi3D.cpp
  gxf/extensions/bi3d/inference/Bi3D.h
  gxf/extensions/bi3d/inference/Bi3DPreProcessor.cpp
  gxf/extensions/bi3d/inference/Bi3DPostProcessor.cpp

  # Inferencer (Bi3D only)
  gxf/gems/dnn_inferencer/inferencer/TensorRTInferencer.cpp
  gxf/gems/dnn_inferencer/inferencer/TensorRTUtils.cpp
  gxf/gems/dnn_inferencer/inferencer/Inferencer.cpp
  gxf/gems/dnn_inferencer/inferencer/Errors.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/gxf")

target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
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

set_target_properties(${PROJECT_NAME} PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Install the binary file
install(TARGETS ${PROJECT_NAME} DESTINATION share/${PROJECT_NAME}/gxf/lib)


# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE)
