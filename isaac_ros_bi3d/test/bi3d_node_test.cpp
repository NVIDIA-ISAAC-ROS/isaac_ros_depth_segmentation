// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gmock/gmock.h>
#include "bi3d_node.hpp"
#include "rclcpp/rclcpp.hpp"

// Objective: to cover code lines where exceptions are thrown
// Approach: send Invalid Arguments for node parameters to trigger the exception


TEST(bi3d_node_test, test_featnet_engine_file_path)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("featnet_engine_file_path", "");
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::bi3d::Bi3DNode bi3d_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Empty featnet_engine_file_path"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(bi3d_node_test, test_segnet_engine_file_path)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("featnet_engine_file_path", "path_to_featnet_engine");
  options.append_parameter_override("segnet_engine_file_path", "");
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::bi3d::Bi3DNode bi3d_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Empty segnet_engine_file_path"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(bi3d_node_test, test_disparity_values)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("featnet_engine_file_path", "path_to_featnet_engine");
  options.append_parameter_override("segnet_engine_file_path", "path_to_segnet_engine");
  options.append_parameter_override("max_disparity_values", 3);
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::bi3d::Bi3DNode bi3d_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Invalid disparity values:"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}


int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
