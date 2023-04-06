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
from launch_ros.actions import ComposableNodeContainer
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
                'max_disparity_values': max_disparity_values}],
        remappings=[
            ('left_image_bi3d', 'infra1/image_rect_raw'),
            ('right_image_bi3d', 'infra2/image_rect_raw')])

    image_format_converter_node_left = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        name='image_format_node_left',
        parameters=[{
                'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('image_raw', 'infra1/image_rect_raw_mono'),
            ('image', 'infra1/image_rect_raw')]
    )

    image_format_converter_node_right = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        name='image_format_node_right',
        parameters=[{
                'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('image_raw', 'infra2/image_rect_raw_mono'),
            ('image', 'infra2/image_rect_raw')]
    )

    pointcloud_node = ComposableNode(
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::PointCloudNode',
        parameters=[{
                'use_color': True,
                'unit_scaling': 1.0
        }],
        remappings=[('left/image_rect_color', 'infra1/image_rect_raw'),
                    ('left/camera_info', 'infra1/camera_info'),
                    ('right/camera_info', 'infra2/camera_info')]
    )

    # RealSense
    realsense_config_file_path = os.path.join(
        get_package_share_directory('isaac_ros_bi3d'),
        'config', 'realsense.yaml'
    )

    realsense_node = ComposableNode(
        package='realsense2_camera',
        plugin='realsense2_camera::RealSenseNodeFactory',
        parameters=[realsense_config_file_path],
        remappings=[
            ('infra1/image_rect_raw', 'infra1/image_rect_raw_mono'),
            ('infra2/image_rect_raw', 'infra2/image_rect_raw_mono')
        ]
    )

    container = ComposableNodeContainer(
        name='bi3d_container',
        namespace='bi3d',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[bi3d_node,
                                      image_format_converter_node_left,
                                      image_format_converter_node_right,
                                      pointcloud_node,
                                      realsense_node],
        output='screen',
        remappings=[
            ('left_image_bi3d', 'infra1/image_rect_raw'),
            ('right_image_bi3d', 'infra2/image_rect_raw')]
    )

    final_launch_description = launch_args + [container]
    return (launch.LaunchDescription(final_launch_description))
