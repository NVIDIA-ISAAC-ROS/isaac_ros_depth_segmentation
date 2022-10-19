# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from launch.actions import DeclareLaunchArgument, ExecuteProcess
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
            'rosbag_path',
            default_value='src/isaac_ros_proximity_segmentation/resources/rosbags/bi3dnode_rosbag',
            description='Path to the rosbag file'),
        DeclareLaunchArgument(
            'max_disparity_values',
            default_value='64',
            description='The maximum number of disparity values given for Bi3D inference'),
        DeclareLaunchArgument(
            'base_link_frame',
            default_value='base_link',
            description='The name of the tf2 frame corresponding to the origin of the robot base'),
        DeclareLaunchArgument(
            'camera_frame',
            default_value='carter_camera_stereo_left',
            description='The name of the tf2 frame corresponding to the camera center'),
        DeclareLaunchArgument(
            'f_x',
            default_value='732.999267578125',
            description='The number of pixels per distance unit in the x dimension'),
        DeclareLaunchArgument(
            'f_y',
            default_value='734.1167602539062',
            description='The number of pixels per distance unit in the y dimension'),
        DeclareLaunchArgument(
            'grid_height',
            default_value='2000',
            description='The desired height of the occupancy grid, in cells'),
        DeclareLaunchArgument(
            'grid_width',
            default_value='2000',
            description='The desired width of the occupancy grid, in cells'),
        DeclareLaunchArgument(
            'grid_resolution',
            default_value='0.01',
            description='The desired resolution of the occupancy grid, in m/cell'),
    ]

    rosbag_path = LaunchConfiguration('rosbag_path')

    # Bi3DNode parameters
    featnet_engine_file_path = LaunchConfiguration('featnet_engine_file_path')
    segnet_engine_file_path = LaunchConfiguration('segnet_engine_file_path')
    max_disparity_values = LaunchConfiguration('max_disparity_values')

    # FreespaceSegmentationNode parameters
    base_link_frame = LaunchConfiguration('base_link_frame')
    camera_frame = LaunchConfiguration('camera_frame')
    f_x_ = LaunchConfiguration('f_x')
    f_y_ = LaunchConfiguration('f_y')
    grid_height = LaunchConfiguration('grid_height')
    grid_width = LaunchConfiguration('grid_width')
    grid_resolution = LaunchConfiguration('grid_resolution')

    bi3d_node = ComposableNode(
        name='bi3d_node',
        package='isaac_ros_bi3d',
        plugin='nvidia::isaac_ros::bi3d::Bi3DNode',
        parameters=[{
                'featnet_engine_file_path': featnet_engine_file_path,
                'segnet_engine_file_path': segnet_engine_file_path,
                'max_disparity_values': max_disparity_values,
                'use_sim_time': True}],
        remappings=[('bi3d_node/bi3d_output', 'bi3d_mask'),
                    ('left_image_bi3d', 'rgb_left'),
                    ('right_image_bi3d', 'rgb_right')]
    )

    freespace_segmentation_node = ComposableNode(
        name='freespace_segmentation_node',
        package='isaac_ros_bi3d_freespace',
        plugin='nvidia::isaac_ros::bi3d_freespace::FreespaceSegmentationNode',
        parameters=[{
            'base_link_frame': base_link_frame,
            'camera_frame': camera_frame,
            'f_x': f_x_,
            'f_y': f_y_,
            'grid_height': grid_height,
            'grid_width': grid_width,
            'grid_resolution': grid_resolution,
            'use_sim_time': True
        }])

    rosbag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', '--loop', rosbag_path],
        output='screen')

    container = ComposableNodeContainer(
        name='bi3d_freespace_container',
        namespace='bi3d_freespace',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            bi3d_node,
            freespace_segmentation_node,
        ],
        output='screen'
    )

    final_launch_description = launch_args + [rosbag_play, container]
    return (launch.LaunchDescription(final_launch_description))
