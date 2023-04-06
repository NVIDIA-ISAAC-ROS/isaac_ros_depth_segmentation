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

import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'featnet_engine_file_path',
            default_value='',
            description='The absolute path to the Bi3D Featnet TensorRT engine plan'),
        DeclareLaunchArgument(
            'segnet_engine_file_path',
            default_value='',
            description='The absolute path to the Bi3D Segnet TensorRT engine plan'),
        DeclareLaunchArgument(
            'max_disparity_values',
            default_value='64',
            description='The maximum number of disparity values given for Bi3D inference'),
    ]

    # Bi3DNode parameters
    featnet_engine_file_path = LaunchConfiguration('featnet_engine_file_path')
    segnet_engine_file_path = LaunchConfiguration('segnet_engine_file_path')
    max_disparity_values = LaunchConfiguration('max_disparity_values')

    bi3d_node = ComposableNode(
        name='bi3d_node',
        package='isaac_ros_bi3d',
        plugin='nvidia::isaac_ros::bi3d::Bi3DNode',
        parameters=[{
                'featnet_engine_file_path': featnet_engine_file_path,
                'segnet_engine_file_path': segnet_engine_file_path,
                'max_disparity_values': max_disparity_values,
                'disparity_values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60]}],
        remappings=[('left_image_bi3d', 'rgb_left'),
                    ('right_image_bi3d', 'rgb_right')])

    pointcloud_node = ComposableNode(
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::PointCloudNode',
        parameters=[{
            'use_color': True,
            'unit_scaling': 1.0
        }],
        remappings=[('left/image_rect_color', 'rgb_left'),
                    ('left/camera_info', 'camera_info_left'),
                    ('right/camera_info', 'camera_info_right'),
                    ('disparity', 'bi3d_node/bi3d_output')])

    container = ComposableNodeContainer(
        name='bi3d_container',
        namespace='bi3d',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[bi3d_node, pointcloud_node],
        output='screen'
    )

    rviz_config_path = os.path.join(get_package_share_directory(
        'isaac_ros_bi3d'), 'config', 'isaac_ros_bi3d_isaac_sim.rviz')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen')

    final_launch_description = launch_args + [container, rviz_node]
    return (launch.LaunchDescription(final_launch_description))
