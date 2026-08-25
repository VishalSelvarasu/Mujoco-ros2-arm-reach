import sys
import os

# Make the project's envs/ package importable. We can't infer this from
# the file's own location reliably, since colcon copies this file into a
# differently-nested install directory than the source tree -- so we use
# an explicit environment variable instead.
project_root = os.environ.get("MUJOCO_ROS2_PROJECT_ROOT")
if not project_root:
    raise RuntimeError(
        "Set MUJOCO_ROS2_PROJECT_ROOT to your project's root directory, e.g.:\n"
        "  export MUJOCO_ROS2_PROJECT_ROOT=~/Downloads/mujoco-ros2-arm-reach"
    )
sys.path.insert(0, project_root)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray

from envs.reach_env import UR5eReachEnv

SIM_RATE_HZ = 50.0


class MujocoSimNode(Node):
    def __init__(self):
        super().__init__("mujoco_sim_node")

        self.env = UR5eReachEnv()
        self.obs, _ = self.env.reset()
        self.last_action = [0.0] * self.env.n_joints

        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.target_pub = self.create_publisher(PoseStamped, "/target_pose", 10)
        self.cmd_sub = self.create_subscription(
            Float64MultiArray, "/arm_joint_commands", self._on_command, 10
        )

        self.timer = self.create_timer(1.0 / SIM_RATE_HZ, self._on_timer)
        # Real UR5e joint names (from the official Menagerie model), not
        # generic placeholders -- lets this line up with what a real
        # driver's /joint_states would actually publish.
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ]

        self.get_logger().info("MuJoCo sim node started, publishing at %.0f Hz" % SIM_RATE_HZ)

    def _on_command(self, msg: Float64MultiArray):
        if len(msg.data) != self.env.n_joints:
            self.get_logger().warn(
                f"Expected {self.env.n_joints} joint commands, got {len(msg.data)}"
            )
            return
        self.last_action = list(msg.data)

    def _on_timer(self):
        self.obs, reward, terminated, truncated, info = self.env.step(self.last_action)

        if terminated or truncated:
            self.get_logger().info(
                f"Episode ended (success={info['success']}, dist={info['distance']:.3f}m). Resetting."
            )
            self.obs, _ = self.env.reset()
            self.last_action = [0.0] * self.env.n_joints

        self._publish_joint_state()
        self._publish_target()

    def _publish_joint_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.env.data.qpos[: self.env.n_joints].tolist()
        msg.velocity = self.env.data.qvel[: self.env.n_joints].tolist()
        self.joint_state_pub.publish(msg)

    def _publish_target(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(self.env.target_pos[0])
        msg.pose.position.y = float(self.env.target_pos[1])
        msg.pose.position.z = float(self.env.target_pos[2])
        # Orientation is unused here (reach task has no orientation goal),
        # but an all-zero quaternion is invalid -- ROS expects a valid
        # unit quaternion, so set identity rotation explicitly.
        msg.pose.orientation.w = 1.0
        self.target_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.env.close()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
