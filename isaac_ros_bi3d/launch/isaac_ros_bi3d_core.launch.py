# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, Dict

from isaac_ros_examples import IsaacROSLaunchFragment
import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


class IsaacROSBi3DLaunchFragment(IsaacROSLaunchFragment):

    @staticmethod
    def get_composable_nodes(interface_specs: Dict[str, Any]) -> Dict[str, ComposableNode]:

        # Bi3DNode parameters
        featnet_engine_file_path = LaunchConfiguration('featnet_engine_file_path')
        segnet_engine_file_path = LaunchConfiguration('segnet_engine_file_path')
        max_disparity_values = LaunchConfiguration('max_disparity_values')

        return {
            'bi3d_node': ComposableNode(
                name='bi3d_node',
                package='isaac_ros_bi3d',
                plugin='nvidia::isaac_ros::bi3d::Bi3DNode',
                parameters=[{
                        'image_height': interface_specs['image_resolution']['height'],
                        'image_width': interface_specs['image_resolution']['width'],
                        'featnet_engine_file_path': featnet_engine_file_path,
                        'segnet_engine_file_path': segnet_engine_file_path,
                        'max_disparity_values': max_disparity_values}],
                remappings=[('left_image_bi3d', 'left/image_rect'),
                            ('right_image_bi3d', 'right/image_rect'),
                            ('left_camera_info_bi3d', 'left/camera_info_rect'),
                            ('right_camera_info_bi3d', 'right/camera_info_rect')]
            )
        }

    def get_launch_actions(interface_specs: Dict[str, Any]) -> \
            Dict[str, launch.actions.OpaqueFunction]:
        return {
            'featnet_engine_file_path': DeclareLaunchArgument(
                'featnet_engine_file_path',
                default_value='',
                description='The absolute path to the Bi3D Featnet TensorRT engine plan'
            ),
            'segnet_engine_file_path': DeclareLaunchArgument(
                'segnet_engine_file_path',
                default_value='',
                description='The absolute path to the Bi3D Segnet TensorRT engine plan'
            ),
            'max_disparity_values': DeclareLaunchArgument(
                'max_disparity_values',
                default_value='64',
                description='The maximum number of disparity values given for Bi3D inference'
            ),
        }

    def generate_launch_description():
        bi3d_container = ComposableNodeContainer(
            package='rclcpp_components',
            name='bi3d_container',
            namespace='',
            executable='component_container_mt',
            composable_node_descriptions=IsaacROSBi3DLaunchFragment
            .get_composable_nodes().values(),
            output='screen'
        )

        return launch.LaunchDescription(
            [bi3d_container] + IsaacROSBi3DLaunchFragment.get_launch_actions().values())
