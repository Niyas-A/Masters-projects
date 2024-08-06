#!/usr/bin/env python3

########
# Author: Anand Kumar, Ben Klingensmith, Niyas Attasseri
# Name: bringup.launch.py
#
# Purpose: RobThePupper. The launch script which exposes all the topics of the mini-pupper robot and activates calibration.
#
# Usage: This has to launched before anything else to bring up the robot.
#        ros2 launch robthepupper bringup.launch.py
#
# Date: 12 June 2024
########

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch.conditions import IfCondition


def launch_bring_up(context, *args, **kwargs):
    robot_name = LaunchConfiguration("robot_name")
    #sim = LaunchConfiguration("sim")
    #rviz = LaunchConfiguration("rviz")
    joint_hardware_connected = LaunchConfiguration("joint_hardware_connected")

    robot_name_str = context.perform_substitution(robot_name)
    description_package = FindPackageShare(f'{robot_name_str}_description')

    description_path = PathJoinSubstitution(
        [description_package, 'urdf', 'mini_pupper_description.urdf.xacro']
    )

    joints_config_path = PathJoinSubstitution(
        [description_package, 'config', 'champ', 'joints.yaml']
    )
    links_config_path = PathJoinSubstitution(
        [description_package, 'config', 'champ', 'links.yaml']
    )
    gait_config_path = PathJoinSubstitution(
        [description_package, 'config', 'champ', 'gait.yaml']
    )

    bringup_launch_path = PathJoinSubstitution(
        [FindPackageShare('champ_bringup'), 'launch', 'bringup.launch.py']
    )

    rviz_config_path = PathJoinSubstitution(
        [description_package, 'rviz', 'urdf_viewer.rviz']
    )

    bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(bringup_launch_path),
        launch_arguments={
            "use_sim_time": "false",
            "robot_name": robot_name,
            "gazebo": "false",
            "rviz": "false",  # set always false to launch RViz2 with costom .rviz file
            "joint_hardware_connected": joint_hardware_connected,
            "publish_foot_contacts": "true",
            "close_loop_odom": "true",
            "joint_controller_topic": "joint_group_effort_controller/joint_trajectory",
            "joints_map_path": joints_config_path,
            "links_map_path": links_config_path,
            "gait_config_path": gait_config_path,
            "description_path": description_path
        }.items(),
    )

    #rviz2_node = Node(
    #    package="rviz2",
    #    namespace="",
    #    executable="rviz2",
    #    name="rviz2",
    #    arguments=["-d", rviz_config_path],
    #    condition=IfCondition(rviz)
    #)
    
    ref_body_pos_node = Node(
        package="mini_pupper_dance",
        namespace="",
        executable="pose_controller",
        name="pose_controller",
        )
    return [ref_body_pos_node,
            bringup_launch]


def generate_launch_description():
    servo_interface_launch_path = PathJoinSubstitution(
        [FindPackageShare('mini_pupper_driver'), 'launch', 'servo_interface.launch.py']
    )
    lidar_launch_path = PathJoinSubstitution(
        [FindPackageShare('mini_pupper_bringup'), 'launch', 'lidar.launch.py']
    )

    joint_hardware_connected = LaunchConfiguration("joint_hardware_connected")

    declare_robot_name = DeclareLaunchArgument(
            name='robot_name',
            default_value='mini_pupper',
            description='Set robot name for multi robot'
        )

    declare_hardware_connected = DeclareLaunchArgument(
            name='joint_hardware_connected',
            default_value='true',
            description='Set to true if connected to a physical robot'
        )

    servo_interface_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(servo_interface_launch_path),
        condition=IfCondition(joint_hardware_connected),
    )

    return LaunchDescription([
        declare_robot_name,
        declare_hardware_connected,
        OpaqueFunction(function=launch_bring_up),
        servo_interface_launch,
    ])
