import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "mujoco_menagerie" / "universal_robots_ur5e" / "scene.xml"

# Workspace bounds for random target sampling (meters, relative to robot base).
# Chosen conservatively so targets are reachable without self-collision,
# based on the UR5e's ~0.85m max reach.
TARGET_BOUNDS = {
    "x": (-0.4, 0.4),
    "y": (-0.4, 0.4),
    "z": (0.15, 0.55),
}

SUCCESS_THRESHOLD = 0.03  # meters; end-effector within 3cm counts as reached
MAX_JOINT_STEP = 0.5      # radians; target offset commanded per RL action
EPISODE_LEN = 200

# Reset joint configuration, found by grid-searching for the pose that puts
# the end-effector closest to the center of TARGET_BOUNDS. The model's
# qpos=0 "home" pose starts ~0.8m away from every target, which starved
# training of useful reward signal early on.
RESET_QPOS = np.array([1.891, 3.766, 2.189, 0.123, 2.034, 1.276])

# The physics timestep is 2ms. Calling mj_step() once per RL action only
# advances 2ms of real time — nowhere near enough for the joint's position
# servo to move meaningfully (empirically: ~0.0002 rad per action instead
# of the intended ~0.05 rad). Running 50 substeps per action gives the
# servo real time to actually respond.
PHYSICS_SUBSTEPS = 50

class UR5eReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._renderer = None

        self.n_joints = self.model.nu  # 6 for UR5e
        self.ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link"
        )

        # Action: normalized delta per joint, in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )

        # Observation: [joint_pos(6), joint_vel(6), ee_pos(3), ee_to_target(3)] = 18
        obs_dim = self.n_joints * 2 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.target_pos = np.zeros(3)
        self.step_count = 0

    def _get_ee_pos(self):
        return self.data.xpos[self.ee_body_id].copy()

    def _get_obs(self):
        ee_pos = self._get_ee_pos()
        return np.concatenate([
            self.data.qpos[: self.n_joints],
            self.data.qvel[: self.n_joints],
            ee_pos,
            self.target_pos - ee_pos,
        ]).astype(np.float32)

    def _sample_target(self):
        return np.array([
            np.random.uniform(*TARGET_BOUNDS["x"]),
            np.random.uniform(*TARGET_BOUNDS["y"]),
            np.random.uniform(*TARGET_BOUNDS["z"]),
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Small random noise around the workspace-centered reset pose.
        self.data.qpos[: self.n_joints] = RESET_QPOS + np.random.uniform(-0.05, 0.05, self.n_joints)
        mujoco.mj_forward(self.model, self.data)

        self.target_pos = self._sample_target()
        self.step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        target_qpos = self.data.qpos[: self.n_joints] + action * MAX_JOINT_STEP
        self.data.ctrl[: self.n_joints] = target_qpos

        # Hold this target fixed across all substeps so the servo has real
        # time to converge toward it before the next action is chosen.
        for _ in range(PHYSICS_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        ee_pos = self._get_ee_pos()
        dist = np.linalg.norm(ee_pos - self.target_pos)

        # Dense shaping + sparse success bonus (see module docstring for why)
        reward = -dist
        success = dist < SUCCESS_THRESHOLD
        if success:
            reward += 10.0

        terminated = bool(success)
        truncated = self.step_count >= EPISODE_LEN

        return self._get_obs(), reward, terminated, truncated, {"distance": dist, "success": success}

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            # Fixed camera angle, closer and more informative than MuJoCo's
            # generic default zoomed-out view.
            self._camera = mujoco.MjvCamera()
            self._camera.distance = 1.3
            self._camera.azimuth = 130
            self._camera.elevation = -20
            self._camera.lookat = np.array([0.0, 0.0, 0.35])

        self._renderer.update_scene(self.data, camera=self._camera)

        # Overlay a marker sphere at the target position -- without this,
        # a viewer has no way to tell what the arm is reaching for, since
        # the target isn't a physical body in the model itself.
        scene = self._renderer.scene
        if scene.ngeom < scene.maxgeom:
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                g,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.025, 0, 0],
                pos=self.target_pos,
                mat=np.eye(3).flatten(),
                rgba=[1.0, 0.2, 0.2, 0.9],
            )
            scene.ngeom += 1

        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()


if __name__ == "__main__":
    # Quick smoke test: random actions, confirm no crashes, print shapes.
    env = UR5eReachEnv()
    obs, info = env.reset()
    print("Observation shape:", obs.shape)
    print("Action space:", env.action_space)
    total_reward = 0
    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward
        if terminated or truncated:
            break
    print("Ran 50 random steps OK. Total reward:", round(total_reward, 3))
    print("Final distance to target:", round(info["distance"], 3))
