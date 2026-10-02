# LIO-SAM in Python

This repository contains a Python implementation of the LIO-SAM package for ROS 2.

## Dependencies

- [Livox ROS Driver](https://github.com/Livox-SDK/livox_ros_driver2) (to use Mid-360 LiDAR)
- Python libraries
  ```bash
  pip3 install -r lio_sam/requirements.txt
  ```

The code has been tested on JetPack 6.2.3 with Python 3.10 and ROS 2 Humble.

## Usage

1. Run Livox Mid-360 LiDAR:
    ```bash
    ros2 launch livox_ros_driver2 msg_MID360_launch.py
    ```

    - Set *xfer_format* to 0 in the launch file to publish PointCloud2 messages.
    - The Livox ROS driver publishes the IMU acceleration as a normalized value.
      To restore the acceleration to units of m/s², multiply each component by standard gravity.

      In *lddc.cpp*, line 493:
        ```cpp
        imu_msg.linear_acceleration.x = imu_data.acc_x * 9.80665;
        imu_msg.linear_acceleration.y = imu_data.acc_y * 9.80665;
        imu_msg.linear_acceleration.z = imu_data.acc_z * 9.80665;
        ```

2. Run LIO-SAM nodes:
    ```bash
    ros2 launch lio_sam run_mid360.launch.py
    ```
    - The Madgwick filter is included in the launch file and is used to estimate the IMU orientation.
