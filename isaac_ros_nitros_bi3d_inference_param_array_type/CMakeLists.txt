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
project(isaac_ros_nitros_bi3d_inference_param_array_type LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# NVTX
option(USE_NVTX "Enable NVTX markers for improved profiling (if available)" ON)
if(USE_NVTX)
    add_definitions(-DUSE_NVTX)
    link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
    link_libraries("nvToolsExt")
endif()

# NitrosBi3DInferenceParamArray
ament_auto_add_library(${PROJECT_NAME} SHARED src/nitros_bi3d_inference_param_array.cpp)

if(BUILD_TESTING)
  # Install test/config directory
  install(DIRECTORY test/config DESTINATION share/${PROJECT_NAME}/test)

  # NitrosBi3DInferenceParamArrayForwardNode
  ament_auto_add_library(nitros_bi3d_inference_param_array_forward_node SHARED test/src/nitros_bi3d_inference_param_array_forward_node.cpp)
  target_link_libraries(nitros_bi3d_inference_param_array_forward_node ${PROJECT_NAME})
  set_target_properties(nitros_bi3d_inference_param_array_forward_node PROPERTIES BUILD_RPATH_USE_ORIGIN TRUE)
  set_target_properties(nitros_bi3d_inference_param_array_forward_node PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
  rclcpp_components_register_nodes(nitros_bi3d_inference_param_array_forward_node
    "nvidia::isaac_ros::nitros::NitrosBi3DInferenceParamArrayForwardNode")
  set(node_plugins "${node_plugins}nvidia::isaac_ros::nitros::NitrosBi3DInferenceParamArrayForwardNode;\
      $<TARGET_FILE:nitros_bi3d_inference_param_array_forward_node>\n")

  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()


  # The FindPythonInterp and FindPythonLibs modules are removed
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_nitros_bi3d_inference_param_array_type_test_pol.py TIMEOUT "15")
endif()

ament_auto_package()
