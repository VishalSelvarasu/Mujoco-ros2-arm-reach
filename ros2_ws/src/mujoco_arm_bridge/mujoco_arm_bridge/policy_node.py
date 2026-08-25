
import sys
import os
import numpy as np
import mujoco
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray

project_root = os.environ.get("MUJOCO_ROS2_PROJECT_ROOT")
if not project_root:
    raise RuntimeError(
        "Set MUJOCO_ROS2_PROJECT_ROOT to your project's root directory, e.g.:\n"
        "  export MUJOCO_ROS2_PROJECT_ROOT=~/Downloads/mujoco-ros2-arm-reach"
    )
sys.path.insert(0, project_root)
from envs.reach_env import MODEL_PATH


class PolicyNode(Node):
    def __init__(self):
        super().__init__("policy_node")

        self.declare_parameter("model_path", "")
        self.declare_parameter("algo", "ppo")

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        algo = self.get_parameter("algo").get_parameter_value().string_value

        if not model_path:
            self.get_logger().error(
                "No model_path parameter provided. Launch with "
                "--ros-args -p model_path:=/path/to/model.zip -p algo:=ppo"
            )
            raise SystemExit(1)

        if algo.lower() == "ppo":
            from stable_baselines3 import PPO as Algo
        elif algo.lower() == "sac":
            from stable_baselines3 import SAC as Algo
        else:
            raise ValueError(f"Unknown algo '{algo}', expected 'ppo' or 'sac'")

        self.model = Algo.load(model_path)
        self.get_logger().info(f"Loaded {algo.upper()} model from {model_path}")

        # Forward-kinematics model, used only to compute end-effector
        # position from joint angles -- not for physics simulation.
        self.fk_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.fk_data = mujoco.MjData(self.fk_model)
        self.ee_body_id = mujoco.mj_name2id(self.fk_model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")

        self.n_joints = 6
        self.joint_pos = np.zeros(self.n_joints)
        self.joint_vel = np.zeros(self.n_joints)
        self.target_pos = np.zeros(3)
        self.have_joint_state = False
        self.have_target = False

        self.cmd_pub = self.create_publisher(Float64MultiArray, "/arm_joint_commands", 10)
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self.create_subscription(PoseStamped, "/target_pose", self._on_target, 10)

        self.timer = self.create_timer(1.0 / 50.0, self._on_timer)

    JOINT_ORDER = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]

    def _on_joint_state(self, msg: JointState):
        # Map by name rather than trusting positional order -- a real
        # robot driver's /joint_states isn't guaranteed to publish in
        # this project's internal joint ordering.
        try:
            idx = [msg.name.index(j) for j in self.JOINT_ORDER]
        except ValueError:
            self.get_logger().warn(f"Unexpected joint names in /joint_states: {msg.name}")
            return
        self.joint_pos = np.array([msg.position[i] for i in idx])
        self.joint_vel = np.array([msg.velocity[i] for i in idx])
        self.have_joint_state = True

    def _on_target(self, msg: PoseStamped):
        self.target_pos = np.array([
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        ])
        self.have_target = True

    def _compute_ee_pos(self):
        self.fk_data.qpos[: self.n_joints] = self.joint_pos
        mujoco.mj_forward(self.fk_model, self.fk_data)
        return self.fk_data.xpos[self.ee_body_id].copy()

    def _on_timer(self):
        if not (self.have_joint_state and self.have_target):
            return

        ee_pos = self._compute_ee_pos()

        obs = np.concatenate([
            self.joint_pos, self.joint_vel, ee_pos,
            self.target_pos - ee_pos,
        ]).astype(np.float32)

        action, _ = self.model.predict(obs, deterministic=True)

        msg = Float64MultiArray()
        msg.data = action.tolist()
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
