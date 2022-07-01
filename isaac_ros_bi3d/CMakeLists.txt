# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

cmake_minimum_required(VERSION 3.8)
project(isaac_ros_bi3d)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Default to Release build
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
message(STATUS "Architecture: ${ARCHITECTURE}")

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# CVCore Bi3D extension library
add_library(cvcore_bi3d SHARED IMPORTED)

if(${ARCHITECTURE} STREQUAL "x86_64")
  set_property(
    TARGET cvcore_bi3d PROPERTY IMPORTED_LOCATION
    ${CMAKE_CURRENT_SOURCE_DIR}/cvcore/cvcore_bi3d/lib/x86_64/libgxf_cvcore_bi3d.so)
elseif(${ARCHITECTURE} STREQUAL "aarch64")
  set_property(
    TARGET cvcore_bi3d PROPERTY IMPORTED_LOCATION
    ${CMAKE_CURRENT_SOURCE_DIR}/cvcore/cvcore_bi3d/lib/jetpack50/libgxf_cvcore_bi3d.so)
endif()

# Isaac ROS Bi3D postprocess extension library
add_library(bi3d_postprocess SHARED IMPORTED)

if(${ARCHITECTURE} STREQUAL "x86_64")
  set_property(
    TARGET bi3d_postprocess PROPERTY IMPORTED_LOCATION
    ${CMAKE_CURRENT_SOURCE_DIR}/cvcore/cvcore_bi3d/lib/x86_64/libgxf_bi3d_postprocess.so)
elseif(${ARCHITECTURE} STREQUAL "aarch64")
  set_property(
    TARGET bi3d_postprocess PROPERTY IMPORTED_LOCATION
    ${CMAKE_CURRENT_SOURCE_DIR}/cvcore/cvcore_bi3d/lib/jetpack50/libgxf_bi3d_postprocess.so)
endif()

# bi3d_node
ament_auto_add_library(bi3d_node SHARED src/bi3d_node.cpp)
target_compile_definitions(bi3d_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
target_link_libraries(bi3d_node cvcore_bi3d bi3d_postprocess)
rclcpp_components_register_nodes(bi3d_node "nvidia::isaac_ros::bi3d::Bi3DNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::bi3d::Bi3DNode;$<TARGET_FILE:bi3d_node>\n")

# Install config directory
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# Install extensions directory
if(${ARCHITECTURE} STREQUAL "x86_64")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cvcore/cvcore_bi3d/lib/x86_64/ DESTINATION share/${PROJECT_NAME}/cvcore_bi3d)
elseif(${ARCHITECTURE} STREQUAL "aarch64")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cvcore/cvcore_bi3d/lib/jetpack50/ DESTINATION share/${PROJECT_NAME}/cvcore_bi3d)
endif()

set_target_properties(bi3d_node PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
install(TARGETS bi3d_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  # Ignore copyright notices since we use custom NVIDIA Isaac ROS Software License
  set(ament_cmake_copyright_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_bi3d_test.py TIMEOUT "1000")
endif()

install(PROGRAMS
  scripts/isaac_ros_bi3d_visualizer.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package(INSTALL_TO_SHARE launch)
