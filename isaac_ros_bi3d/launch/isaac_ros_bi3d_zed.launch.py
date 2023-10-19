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
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    # The zed camera model name. zed, zed2, zed2i, zedm, zedx or zedxm
    camera_model = 'zed2i'

    launch_args = [
        DeclareLaunchArgument(
            'image_height',
            default_value='1080',
            description='The height of the input image'),
        DeclareLaunchArgument(
            'image_width',
            default_value='3840',
            description='The width of the input image'),
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
    image_height = LaunchConfiguration('image_height')
    image_width = LaunchConfiguration('image_width')
    featnet_engine_file_path = LaunchConfiguration('featnet_engine_file_path')
    segnet_engine_file_path = LaunchConfiguration('segnet_engine_file_path')
    max_disparity_values = LaunchConfiguration('max_disparity_values')

    bi3d_node = ComposableNode(
        name='bi3d_node',
        package='isaac_ros_bi3d',
        plugin='nvidia::isaac_ros::bi3d::Bi3DNode',
        parameters=[{
                'image_height': image_height,
                'image_width': image_width,
                'featnet_engine_file_path': featnet_engine_file_path,
                'segnet_engine_file_path': segnet_engine_file_path,
                'max_disparity_values': max_disparity_values}],
        remappings=[
            ('left_image_bi3d', 'zed_node/left/image_rect_color_rgb'),
            ('right_image_bi3d', 'zed_node/right/image_rect_color_rgb'),
            ('left_camera_info_bi3d', 'zed_node/left/camera_info'),
            ('right_camera_info_bi3d', 'zed_node/right/camera_info')]
    )

    pointcloud_node = ComposableNode(
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::PointCloudNode',
        parameters=[{
                'use_color': True,
                'unit_scaling': 1.0,
                'output_height': image_height,
                'output_width': image_width
        }],
        remappings=[('left/image_rect_color', 'zed_node/left/image_rect_color_rgb'),
                    ('left/camera_info', 'zed_node/left/camera_info'),
                    ('right/camera_info', 'zed_node/right/camera_info'),
                    ('disparity', 'bi3d_node/bi3d_output')]
    )

    image_format_converter_node_left = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        name='image_format_node_left',
        parameters=[{
                'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('image_raw', 'zed_node/left/image_rect_color'),
            ('image', 'zed_node/left/image_rect_color_rgb')]
    )

    image_format_converter_node_right = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        name='image_format_node_right',
        parameters=[{
                'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('image_raw', 'zed_node/right/image_rect_color'),
            ('image', 'zed_node/right/image_rect_color_rgb')]
    )

    container = ComposableNodeContainer(
        name='bi3d_container',
        package='rclcpp_components',
        namespace='',
        executable='component_container_mt',
        composable_node_descriptions=[bi3d_node,
                                      image_format_converter_node_left,
                                      image_format_converter_node_right,
                                      pointcloud_node],
        output='screen',
        remappings=[
            ('left_image_bi3d', 'zed_node/left/image_rect_color'),
            ('right_image_bi3d', 'zed_node/right/image_rect_color')]
    )

    rviz_config_path = os.path.join(get_package_share_directory(
        'isaac_ros_bi3d'), 'config', 'isaac_ros_bi3d_zed.rviz')

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen')

    # URDF/xacro file to be loaded by the Robot State Publisher node
    xacro_path = os.path.join(
        get_package_share_directory('zed_wrapper'),
        'urdf', 'zed_descr.urdf.xacro'
    )

    # ZED Configurations to be loaded by ZED Node
    config_common = os.path.join(
        get_package_share_directory('isaac_ros_bi3d'),
        'config',
        'zed.yaml'
    )

    config_camera = os.path.join(
        get_package_share_directory('zed_wrapper'),
        'config',
        camera_model + '.yaml'
    )

    # Robot State Publisher node
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='zed_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': Command(
                [
                    'xacro', ' ', xacro_path, ' ',
                    'camera_name:=', camera_model, ' ',
                    'camera_model:=', camera_model
                ])
        }]
    )

    # ZED node using manual composition
    zed_node = Node(
        package='zed_wrapper',
        executable='zed_wrapper',
        output='screen',
        parameters=[
            config_common,  # Common parameters
            config_camera,  # Camera related parameters
        ]
    )

    # Add nodes and containers to LaunchDescription
    return launch.LaunchDescription(launch_args + [container, rviz, rsp_node, zed_node])
