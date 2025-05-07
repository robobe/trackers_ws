from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    video_path_arg = DeclareLaunchArgument(
        'video_path',
        default_value='default.mp4',
        description='Path to the video file'
    )
    
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='30.0',
        description='Frame publish rate in Hz'
    )

    # Create node actions
    video_player_node = Node(
        package='tracker_tester',
        executable='play_video.py',
        # name='video_player',
        # parameters=[{
        #     'video_path': LaunchConfiguration('video_path'),
        #     'publish_rate': LaunchConfiguration('publish_rate')
        # }]
    )

    tracker_node = Node(
        package='tracker_nano',
        executable='tracker.py',
        name='tracker'
    )

    viewer_node = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='image_viewer'
    )

    return LaunchDescription([
        video_path_arg,
        publish_rate_arg,
        video_player_node,
        tracker_node
        # viewer_node
    ])