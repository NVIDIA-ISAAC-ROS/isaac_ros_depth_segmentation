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

#ifndef ISAAC_ROS_BI3D__BI3D_NODE_HPP_
#define ISAAC_ROS_BI3D__BI3D_NODE_HPP_

#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/rclcpp/parameter_event_handler.hpp"
#include "isaac_ros_nitros/nitros_node.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace bi3d
{

class Bi3DNode : public nitros::NitrosNode
{
public:
  explicit Bi3DNode(const rclcpp::NodeOptions &);

  ~Bi3DNode();

  Bi3DNode(const Bi3DNode &) = delete;

  Bi3DNode & operator=(const Bi3DNode &) = delete;

  // The callback to be implemented by users for any required initialization
  void preLoadGraphCallback() override;
  void postLoadGraphCallback() override;

  // Callback for setting correct VideoBuffer name
  void Bi3DVideoBufferNameCallback(
    const gxf_context_t, nitros::NitrosTypeBase &, std::string name, bool pub_disp_values);

private:
  // Bi3D model input paramters
  const std::string featnet_engine_file_path_;
  const std::vector<std::string> featnet_input_layers_name_;
  const std::vector<std::string> featnet_output_layers_name_;

  const std::string segnet_engine_file_path_;
  const std::vector<std::string> segnet_input_layers_name_;
  const std::vector<std::string> segnet_output_layers_name_;

  // Bi3D extra parameters
  int64_t max_disparity_values_;
  std::vector<int64_t> disparity_values_;

  // Dynamic parameter callbacks and handles
  OnSetParametersCallbackHandle::SharedPtr on_set_param_cb_handle_{nullptr};
};

}  // namespace bi3d
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_BI3D__BI3D_NODE_HPP_
