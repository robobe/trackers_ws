import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node

PKG = "tracker_nano"

def generate_launch_description():
    ld = LaunchDescription()
    
    params = PathJoinSubstitution([
        get_package_share_directory(PKG),
        'config',
        "nano.yaml"
    ])

    node = Node(
            package='tracker_nano',
            executable='tracker.py',
            output='screen',
            parameters=[params])

    ld.add_action(node)
    return ld
