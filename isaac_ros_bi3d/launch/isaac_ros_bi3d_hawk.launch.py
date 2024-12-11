# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        DeclareLaunchArgument(
            'module_id',
            default_value='2',
            description='Index specifying the stereo camera module to use.'),
    ]

    module_id = LaunchConfiguration('module_id')

    # Bi3DNode parameters
    featnet_engine_file_path = LaunchConfiguration('featnet_engine_file_path')
    segnet_engine_file_path = LaunchConfiguration('segnet_engine_file_path')
    max_disparity_values = LaunchConfiguration('max_disparity_values')

    argus_stereo_node = ComposableNode(
        name='argus_stereo',
        package='isaac_ros_argus_camera',
        plugin='nvidia::isaac_ros::argus::ArgusStereoNode',
        parameters=[{
            'module_id': module_id
        }],
    )

    left_resize_node = ComposableNode(
        name='left_resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': 960,
            'output_height': 576,
            'keep_aspect_ratio': True
        }],
        remappings=[
            ('camera_info', '/left/camera_info'),
            ('image', '/left/image_raw'),
            ('resize/camera_info', '/left/camera_info_resize'),
            ('resize/image', '/left/image_resize')]
    )

    right_resize_node = ComposableNode(
        name='right_resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': 960,
            'output_height': 576,
            'keep_aspect_ratio': True
        }],
        remappings=[
            ('camera_info', '/right/camera_info'),
            ('image', '/right/image_raw'),
            ('resize/camera_info', '/right/camera_info_resize'),
            ('resize/image', '/right/image_resize')]
    )

    left_rectify_node = ComposableNode(
        name='left_rectify_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        parameters=[{
            'output_width': 960,
            'output_height': 576,
        }],
        remappings=[
            ('image_raw', '/left/image_resize'),
            ('camera_info', '/left/camera_info_resize'),
            ('image_rect', '/left/image_rect'),
            ('camera_info_rect', '/left/camera_info_rect')
        ]
    )

    right_rectify_node = ComposableNode(
        name='right_rectify_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        parameters=[{
            'output_width': 960,
            'output_height': 576,
        }],
        remappings=[
            ('image_raw', '/right/image_resize'),
            ('camera_info', '/right/camera_info_resize'),
            ('image_rect', '/right/image_rect'),
            ('camera_info_rect', '/right/camera_info_rect')
        ]
    )

    bi3d_node = ComposableNode(
        name='bi3d_node',
        package='isaac_ros_bi3d',
        plugin='nvidia::isaac_ros::bi3d::Bi3DNode',
        parameters=[{
                'featnet_engine_file_path': featnet_engine_file_path,
                'segnet_engine_file_path': segnet_engine_file_path,
                'max_disparity_values': max_disparity_values,
                'image_width': 960,
                'image_height': 576
                }],
        remappings=[('bi3d_node/bi3d_output', 'bi3d_mask'),
                    ('left_image_bi3d', 'left/image_rect'),
                    ('left_camera_info_bi3d', 'left/camera_info_rect'),
                    ('right_image_bi3d', 'right/image_rect'),
                    ('right_camera_info_bi3d', 'right/camera_info_rect')]
    )

    bi3d_launch_container = ComposableNodeContainer(
        name='bi3d_launch_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            argus_stereo_node, left_resize_node, right_resize_node,
            left_rectify_node, right_rectify_node, bi3d_node
        ],
        output='screen'
    )

    return (launch.LaunchDescription(launch_args + [bi3d_launch_container]))
