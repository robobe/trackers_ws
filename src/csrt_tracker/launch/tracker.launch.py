import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node

PKG = "csrt_tracker"

def generate_launch_description():
    ld = LaunchDescription()

    config_file = PathJoinSubstitution([
        get_package_share_directory(PKG),
        'config',
        'tracker.yaml'
    ])

    node = Node(
        package=PKG,
        executable='tracker.py',
        name='tracker',
        output='screen',
        parameters=[config_file])

    ld.add_action(node)
    return ld