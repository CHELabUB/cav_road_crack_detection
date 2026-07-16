from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource, AnyLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # RTK GPS
    novatel_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('novatel_oem7_driver'),
                'launch',
                'oem7_net.launch.py'
            )
        ),
        launch_arguments={
            'oem7_ip_addr': '192.168.100.201',
            'oem7_port': '3005'
        }.items()
    )

    # DBW
    dbw_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ds_dbw_can'),
                'launch',
                'dbw.launch.xml'
            )
        )
    )

    # Mako camera
    mako_camera = Node(
        package='vimbax_camera',
        executable='vimbax_camera_node',
        name='mako_camera',
        output='screen',
        parameters=[{'use_ros_time': True}]
    )

    # Paths to the on-vehicle, camera, and detection node scripts
    nodes_dir = os.path.join('/ws/cv2x/detection_pipeline', 'nodes') # this path needs to be updated, as it affects the logs saving
    obu_node_path = os.path.join(nodes_dir, 'obu_node.py')
    on_vehicle_path = os.path.join(nodes_dir, 'vehicle_node.py')
    camera_node_path = os.path.join(nodes_dir, 'camera_node.py')
    detection_node_path = os.path.join(nodes_dir, 'detection_node.py')

    # vehicle node
    on_vehicle_node = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=['python3', on_vehicle_path, '--data-collection'],
                output='screen'
            )
        ]
    )

    # OBU node
    obu_node = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=['python3', obu_node_path],
                output='screen'
            )
        ]
    )

    # Camera node
    camera_node = TimerAction(
        period=2.0,  # Delay of 2 seconds
        actions=[
            ExecuteProcess(
                cmd=['python3', camera_node_path],
                output='screen',
            )
        ]
    )

    # Detection node
    detection_node = TimerAction(
        period=1.0,
        actions=[
            ExecuteProcess(
                cmd=['python3', detection_node_path],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        novatel_launch,
        dbw_launch,
        mako_camera,
        on_vehicle_node,
        obu_node,
        camera_node,
        detection_node,

    ])
