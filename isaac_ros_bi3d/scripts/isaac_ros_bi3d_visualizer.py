#!/usr/bin/env python3

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

# This script plays a rosbag with prerecorded images and sends them to the Bi3DNode for inference,
# then either saves the output prediction to spcified location as an image, or displays
# it to the screen

import argparse
import subprocess

import cv2
import cv_bridge
import rclpy
from rclpy.node import Node
from stereo_msgs.msg import DisparityImage


def get_args():
    parser = argparse.ArgumentParser(description='Bi3D Node Visualizer')
    parser.add_argument('--save_image', action='store_true', help='Save output or display it',
                        default=False)
    parser.add_argument('--enable_rosbag', action='store_true', help='Save output or display it',
                        default=False)
    parser.add_argument('--max_disparity_value', type=int,
                        help='Maximium disparity value given to Bi3D Node', default=18)
    parser.add_argument('--result_path', default='/workspaces/isaac_ros-dev/src/bi3d_output.png',
                        help='Absolute path to save your result.')
    parser.add_argument('--disparity_topic', default='bi3d_node/bi3d_output',
                        help='Topic name for disparity output.')
    parser.add_argument('--rosbag_path',
                        default='/workspaces/isaac_ros-dev/src/'
                                'isaac_ros_proximity_segmentation/resources/'
                                'rosbags/bi3dnode_rosbag',
                        help='Absolute path to your rosbag.')
    args = parser.parse_args()
    return args


class Bi3DVisualizer(Node):

    def __init__(self, args):
        super().__init__('bi3d_visualizer')
        self.args = args
        self.encoding = 'rgb8'
        self._bridge = cv_bridge.CvBridge()
        self._bi3d_sub = self.create_subscription(
            DisparityImage, self.args.disparity_topic, self.bi3d_callback, 10)
        if self.args.enable_rosbag:
            subprocess.Popen('ros2 bag play --loop ' + self.args.rosbag_path, shell=True)
        self.image_scale = 255.0

    def bi3d_callback(self, data):
        self.get_logger().info('Receiving Bi3D output')
        bi3d_image = self._bridge.imgmsg_to_cv2(data.image)
        bi3d_image = bi3d_image/self.args.max_disparity_value
        if self.args.save_image:
            cv2.imwrite(self.args.result_path, bi3d_image*self.image_scale)
        else:
            cv2.imshow('bi3d_output', bi3d_image)
        cv2.waitKey(1)


def main():
    args = get_args()
    rclpy.init()
    rclpy.spin(Bi3DVisualizer(args))


if __name__ == '__main__':
    main()
