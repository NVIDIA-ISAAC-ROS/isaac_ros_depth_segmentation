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

import time

from isaac_ros_bi3d_interfaces.msg import Bi3DInferenceParametersArray
from isaac_ros_test import IsaacROSBaseTest

import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import launch_testing
import pytest
import rclpy


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with NitrosNode test node."""
    test_ns = IsaacROSNitrosBi3DInferenceParamArrayTest.generate_namespace()
    container = ComposableNodeContainer(
        name='image_container',
        namespace='isaac_ros_nitros_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='isaac_ros_nitros_bi3d_inference_param_array_type',
                plugin='nvidia::isaac_ros::nitros::NitrosBi3DInferenceParamArrayForwardNode',
                name='NitrosBi3DInferenceParamArrayForwardNode',
                namespace=test_ns,
                parameters=[{
                    'compatible_format': 'nitros_bi3d_inference_param_array'
                }]
            ),
        ],
        output='both',
        arguments=['--ros-args', '--log-level', 'info'],
    )

    return IsaacROSNitrosBi3DInferenceParamArrayTest.generate_test_description([
        container,
        launch.actions.TimerAction(
            period=2.5, actions=[launch_testing.actions.ReadyToTest()])
    ])


class IsaacROSNitrosBi3DInferenceParamArrayTest(IsaacROSBaseTest):
    """
    Proof-of-Life Test for Isaac ROS Nitros Node.

    1. Sets up ROS publisher to send Bi3DInferenceParametersArray values
    2. Sets up ROS subscriber to listen to output channel of NitrosNode
    3. Verify received messages
    """

    def test_forward_node(self) -> None:
        self.node._logger.info('Starting Isaac ROS NitrosNode POL Test')

        # Subscriber
        received_messages = {}

        subscriber_topic_namespace = self.generate_namespace('topic_forward_output')
        test_subscribers = [
            (subscriber_topic_namespace, Bi3DInferenceParametersArray)
        ]

        subs = self.create_logging_subscribers(
            subscription_requests=test_subscribers,
            received_messages=received_messages,
            use_namespace_lookup=False,
            accept_multiple_messages=True,
            add_received_message_timestamps=True
        )

        # Publisher
        publisher_topic_namespace = self.generate_namespace('topic_forward_input')
        pub = self.node.create_publisher(
            Bi3DInferenceParametersArray,
            publisher_topic_namespace,
            self.DEFAULT_QOS)

        try:
            # Construct Bi3D inference parameters array message
            INPUT_DATA = [1, 2, 3]
            bi3d_inf_params = Bi3DInferenceParametersArray()
            bi3d_inf_params.disparity_values = INPUT_DATA

            # Start sending messages
            self.node.get_logger().info('Start publishing messages')
            sent_count = 0
            end_time = time.time() + 2.0
            while time.time() < end_time:
                sent_count += 1
                pub.publish(bi3d_inf_params)
                rclpy.spin_once(self.node, timeout_sec=0.2)

            # Conclude the test
            received_count = len(received_messages[subscriber_topic_namespace])
            self.node._logger.info(
                f'Test Results:\n'
                f'# of Messages Sent: {sent_count}\n'
                f'# of Messages Received: {received_count}\n'
                f'# of Messages Dropped: {sent_count - received_count}\n'
                f'Message Drop Rate: {((sent_count-received_count)/sent_count)*100}%'
            )

            self.assertGreater(len(received_messages[subscriber_topic_namespace]), 0)
            for i in range(len(INPUT_DATA)):
                self.assertEqual(
                    received_messages[subscriber_topic_namespace][-1][0].disparity_values[i],
                    bi3d_inf_params.disparity_values[i])

            self.node._logger.info('Source and received messages are matched.')

        finally:
            [self.node.destroy_subscription(sub) for sub in subs]
            self.assertTrue(self.node.destroy_publisher(pub))
