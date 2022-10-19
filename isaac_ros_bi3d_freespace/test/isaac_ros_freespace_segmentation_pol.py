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

import time

from isaac_ros_test import IsaacROSBaseTest

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from nav_msgs.msg import OccupancyGrid
import pytest
import rclpy
from stereo_msgs.msg import DisparityImage


@pytest.mark.rostest
def generate_test_description():
    freespace_segmentation_node = ComposableNode(
        name='freespace_segmentation',
        package='isaac_ros_bi3d_freespace',
        plugin='nvidia::isaac_ros::bi3d_freespace::FreespaceSegmentationNode',
        namespace=IsaacROSFreespaceSegmentationTest.generate_namespace(),
        parameters=[{
            'base_link_frame': 'base_link',
            'camera_frame': 'camera',
            'f_x': 2000.0,
            'f_y': 2000.0,
            'grid_width': 200,
            'grid_height': 100,
            'grid_resolution': 0.01
        }]
    )

    tf_publisher = ComposableNode(
        name='static_transform_publisher',
        package='tf2_ros',
        plugin='tf2_ros::StaticTransformBroadcasterNode',
        parameters=[{
            'frame_id': 'base_link',
            'child_frame_id': 'camera',
            'translation.x': -0.3,
            'translation.y': 0.2,
            'translation.z': 0.5,
            'rotation.x': -0.12,
            'rotation.y': 0.98,
            'rotation.z': -0.17,
            'rotation.w': 0.02,
        }]
    )

    container = ComposableNodeContainer(
        name='freespace_segmentation_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[freespace_segmentation_node, tf_publisher],
        output='screen',
        arguments=['--ros-args', '--log-level', 'info']
    )
    return IsaacROSFreespaceSegmentationTest.generate_test_description([container])


class IsaacROSFreespaceSegmentationTest(IsaacROSBaseTest):
    IMAGE_HEIGHT = 576
    IMAGE_WIDTH = 960
    TIMEOUT = 1000
    GXF_WAIT_SEC = 3

    def _create_disparity_image(self, name):

        disp_image = DisparityImage()
        disp_image.image.height = self.IMAGE_HEIGHT
        disp_image.image.width = self.IMAGE_WIDTH
        disp_image.image.encoding = '32FC1'
        disp_image.image.is_bigendian = False
        disp_image.image.step = self.IMAGE_WIDTH * 4

        """
        Creates the following test pattern:

        ###             ###


        #
        #
        ###################

        """

        data = []
        for row in range(self.IMAGE_HEIGHT):
            if row == 0:
                data += [0] * self.IMAGE_WIDTH + [1] * self.IMAGE_WIDTH + \
                    [1] * self.IMAGE_WIDTH + [0] * self.IMAGE_WIDTH
            elif row == 575:
                data += [0] * self.IMAGE_WIDTH + [0] * self.IMAGE_WIDTH + \
                    [0] * self.IMAGE_WIDTH + [0] * self.IMAGE_WIDTH
            elif row > 400:
                data += [0] * 4 + [1] * (self.IMAGE_WIDTH - 1) * 4
            else:
                data += [1] * self.IMAGE_WIDTH * 4

        disp_image.image.data = data

        disp_image.header.frame_id = name
        return disp_image

    def test_pol(self):
        time.sleep(self.GXF_WAIT_SEC)

        received_messages = {}

        self.generate_namespace_lookup(['bi3d_mask', 'freespace_segmentation/occupancy_grid'])

        disparity_image_pub = self.node.create_publisher(
            DisparityImage, self.namespaces['bi3d_mask'], self.DEFAULT_QOS
        )

        subs = self.create_logging_subscribers(
            [('freespace_segmentation/occupancy_grid', OccupancyGrid)], received_messages)

        try:
            disparity_image = self._create_disparity_image('camera')

            end_time = time.time() + self.TIMEOUT
            done = False

            while time.time() < end_time:
                disparity_image_pub.publish(disparity_image)

                rclpy.spin_once(self.node, timeout_sec=0.1)

                if 'freespace_segmentation/occupancy_grid' in received_messages:
                    done = True
                    break
            self.assertTrue(
                done, "Didn't recieve output on freespace_segmentation/occupancy_grid topic")

            occupancy_grid = received_messages['freespace_segmentation/occupancy_grid']
            self.assertTrue(0 in occupancy_grid.data, 'No cells were marked as free!')

        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(disparity_image_pub)
