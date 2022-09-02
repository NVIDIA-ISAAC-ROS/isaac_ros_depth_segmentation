/**
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef ISAAC_ROS_BI3D__BI3D_NODE_HPP_
#define ISAAC_ROS_BI3D__BI3D_NODE_HPP_

#include <string>
#include <chrono>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
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
  void Bi3DVideoBufferNameCallback(const gxf_context_t, nitros::NitrosTypeBase &, std::string name);

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
};

}  // namespace bi3d
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_BI3D__BI3D_NODE_HPP_
