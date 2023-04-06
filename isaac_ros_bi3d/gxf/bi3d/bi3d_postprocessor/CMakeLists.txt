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
project(gxf_bi3d_postprocessor LANGUAGES C CXX)

# Create extension
add_library(gxf_bi3d_postprocessor SHARED
  bi3d_postprocessor_extension.cpp
  bi3d_postprocess/bi3d_message_splitter.cpp
  bi3d_postprocess/bi3d_message_splitter.hpp
  bi3d_postprocessor_utils.hpp
  bi3d_postprocess/bi3d_postprocessor.cpp
  bi3d_postprocess/bi3d_postprocessor.cu.cpp
  bi3d_postprocess/bi3d_postprocessor.cu.hpp
  bi3d_postprocess/bi3d_postprocessor.hpp
)

# Mark as CUDA files with non-standard extensions
set_source_files_properties(
  bi3d_postprocess/bi3d_postprocessor.cu.cpp
  bi3d_postprocess/bi3d_postprocessor.cu.hpp
  PROPERTIES
    LANGUAGE CUDA
)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

target_link_libraries(gxf_bi3d_postprocessor
  PUBLIC
    GXF::cuda
    GXF::multimedia
    GXF::std
    yaml-cpp
)
