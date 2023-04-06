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

cmake_minimum_required(VERSION 3.23.2)
project(isaac_ros_bi3d LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Bi3DNode
ament_auto_add_library(bi3d_node SHARED src/bi3d_node.cpp)
rclcpp_components_register_nodes(bi3d_node "nvidia::isaac_ros::bi3d::Bi3DNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::bi3d::Bi3DNode;$<TARGET_FILE:bi3d_node>\n")

### Install extensions built from source

# Bi3d
add_subdirectory(gxf/bi3d)
install(TARGETS gxf_cvcore_bi3d DESTINATION share/${PROJECT_NAME}/gxf/lib/bi3d)

# Bi3d postprocessor
add_subdirectory(gxf/bi3d_postprocessor)
install(TARGETS gxf_bi3d_postprocessor DESTINATION share/${PROJECT_NAME}/gxf/lib/bi3d)

### End extensions

install(PROGRAMS scripts/isaac_ros_bi3d_visualizer.py DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_bi3d_test.py TIMEOUT "1000")
endif()

ament_auto_package(INSTALL_TO_SHARE launch config)
