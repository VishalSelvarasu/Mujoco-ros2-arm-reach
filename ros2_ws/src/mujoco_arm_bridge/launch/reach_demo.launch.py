from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_path_arg = DeclareLaunchArgument(
        "model_path", default_value="", description="Path to trained .zip model"
    )
    algo_arg = DeclareLaunchArgument(
        "algo", default_value="ppo", description="'ppo' or 'sac'"
    )

    sim_node = Node(
        package="mujoco_arm_bridge",
        executable="sim_node",
        name="mujoco_sim_node",
        output="screen",
    )

    policy_node = Node(
        package="mujoco_arm_bridge",
        executable="policy_node",
        name="policy_node",
        output="screen",
        parameters=[{
            "model_path": LaunchConfiguration("model_path"),
            "algo": LaunchConfiguration("algo"),
        }],
    )

    return LaunchDescription([model_path_arg, algo_arg, sim_node, policy_node])
