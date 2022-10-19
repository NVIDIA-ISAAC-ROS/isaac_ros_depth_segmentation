// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#ifndef ISAAC_ROS_BI3D_FREESPACE__FREESPACE_SEGMENTATION_NODE_HPP_
#define ISAAC_ROS_BI3D_FREESPACE__FREESPACE_SEGMENTATION_NODE_HPP_

#include <string>
#include <chrono>
#include <utility>
#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_nitros/nitros_node.hpp"

#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"

namespace nvidia
{
namespace isaac_ros
{
namespace bi3d_freespace
{

class FreespaceSegmentationNode : public nitros::NitrosNode
{
public:
  explicit FreespaceSegmentationNode(const rclcpp::NodeOptions &);
  ~FreespaceSegmentationNode();

  void postLoadGraphCallback() override;

private:
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> transform_listener_;

  std::string base_link_frame_;
  std::string camera_frame_;
  std::vector<double> projection_transform_param_;

  double f_x_{};
  double f_y_{};
  std::vector<double> intrinsics_param_;

  int grid_height_;
  int grid_width_;
  double grid_resolution_;
};

}  // namespace bi3d_freespace
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_BI3D_FREESPACE__FREESPACE_SEGMENTATION_NODE_HPP_
