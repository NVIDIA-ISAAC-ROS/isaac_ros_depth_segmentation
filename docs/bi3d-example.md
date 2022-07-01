# Example Usage of Isaac ROS Proximity Segmentation

This is an example use case of Isaac ROS Proximity Segmentation, which includes monitoring the forward zones of an autonomous mobile robot (AMR) using a HAWK stereo camera. Consider three zones in front of the robot (Zones 1-3), corresponding to different alerts and actions the robot must take if an object breaches the zone. Zone 3 generates a warning, Zone 2 sends a signal for the robot to slow, and Zone 1 results in the robot stopping immediately. 

<div align="center"><img src="../resources/safety_zones.png" width="400px"/></div>

Depending on the velocity of the robot, the distances of the zones may change. For this example, we use the following distances from the front of robot to define the zones:

| Robot Velocity | Zone 1 (Stop - Red) | Zone 2 (Slow - Yellow) | Zone 3 (Warning - Blue) |
| -------------- | ------------------- | ---------------------- | ----------------------- |
| 1 m/s          | 0.5 m               | 0.7 m                  | 4.0 m                   |
| 2 m/s          | 2.0 m               | 4.0 m                  | 6.0 m                   |
| 3 m/s          | 3.5 m               | 6.0 m                  | 9.0 m                   |

The distances for each zone is converted to disparity values using the following formula:

<div align="center"><img src="../resources/disparity_equation.png" width="400px"/></div>

This example uses a HAWK stereo camera with a baseline of 15 cm and focal length of 933 px. Thus, the following disparity values are calculated for each zone:

| Robot Velocity | Zone 1 (Stop - Red) | Zone 2 (Slow - Yellow) | Zone 3 (Warning - Blue) |
| -------------- | ------------------- | ---------------------- | ----------------------- |
| 1 m/s          | 280 px              | 200 px                 | 46 px                   |
| 2 m/s          | 70 px               | 35 px                  | 23 px                   |
| 3 m/s          | 40 px               | 23 px                  | 15 px                   |

This example uses the Isaac ROS Bi3D package to detect when these zones are breached. An example is shown in the table below. Pixels in red indicate that Zone 1 has been breached, yellow indicates Zone 2 has been breached, and blue indicates Zone 3 has been breached.

| Input Scene                                                                     | 1 m/s Zones                                                                            | 2 m/s Zones                                                                            | 3 m/s Zones                                                                            |
| ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| <div align="center"><img src="../resources/bi3d_left.png" width="300px"/></div> | <div align="center"><img src="../resources/bi3d_output_1mps.png" width="300px"/></div> | <div align="center"><img src="../resources/bi3d_output_2mps.png" width="300px"/></div> | <div align="center"><img src="../resources/bi3d_output_3mps.png" width="300px"/></div> |
