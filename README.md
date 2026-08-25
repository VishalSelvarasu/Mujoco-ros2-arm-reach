# UR5e Reach: MuJoCo RL Policy Deployed via ROS 2

A UR5e arm learns to reach randomly placed 3D targets in MuJoCo via reinforcement learning, then gets deployed through a ROS 2 node interface. I built this to compare PPO against SAC on a continuous control task, and to get past the part most RL tutorials stop short of: taking a trained policy out of a script and wiring it into a ROS 2 topic interface.

## Why a reach task, not full grasping

Full grasping (arm plus gripper contact dynamics) was the obvious next step after doing navigation work, but contact-rich reward shaping is notoriously slow to get right — it can eat weeks before anything trains reliably. A pure reach task skips the contact problem while still exercising the whole pipeline: environment design, training, and ROS 2 deployment.

I used the official [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) UR5e model instead of building a URDF from scratch. No reason to reinvent a robot model DeepMind already maintains and validates.

## A note on what "reach" measures here

The success condition tracks the position of the `wrist_3_link` body origin, not the model's `attachment_site` (the actual tool-center-point reference, offset about 10cm from that origin in the official model). With a 3cm success threshold, that 10cm gap is not negligible, so I'm calling this a **wrist reference point**, not a true end-effector/TCP reach — the task is real and the numbers below are accurate for what's actually being measured, but it's a wrist-position task, not a precision tool-tip task. Switching to `attachment_site` would be a clean follow-up, at the cost of retraining both models from scratch since it changes the observation and reward computation.

## Repo structure

```
envs/reach_env.py                 - Gymnasium environment wrapping MuJoCo UR5e
training/train_ppo.py             - PPO training with logging
training/train_sac.py             - SAC training with logging
training/compare_results.py       - learning curve plots + head-to-head evaluation
record_demo.py                    - records a GIF of the trained policy in action
models/mujoco_menagerie/          - UR5e MuJoCo model (official, git-cloned, not vendored)
ros2_ws/src/mujoco_arm_bridge/    - ROS 2 package: sim_node + policy_node
results/                          - training outputs land here (gitignored except plots/gif)
```

## Design decisions worth reading

**Actions are normalized joint deltas, not absolute angles.** I tried absolute joint positions first and the policy barely progressed — it had to learn the entire joint range from scratch on every dimension. Small per-step deltas in `[-1, 1]` turn the problem into "which direction do I nudge" instead of "where exactly do I need to be." Converges much faster.

**Reward combines dense distance with a sparse success bonus.** Sparse-only reward gives almost no signal early on, since a random policy essentially never succeeds by chance. Dense-only reward can leave a policy hovering just outside the threshold, because the last few centimeters carry a tiny marginal reward. Using both fixes that.

**The observation includes `target_pos - ee_pos`, not just the raw target.** Handing the policy the relative vector directly saves it from learning that subtraction implicitly. Small change, noticeably faster early training.

**PPO runs 8 parallel environments; SAC runs one.** PPO is on-policy and benefits from diverse experience per update. SAC has a replay buffer, so one environment is enough to keep it fed — running more wouldn't meaningfully help it here.

**Each RL action holds for 50 physics substeps, not one.** The model's physics timestep is 2ms. A single `mj_step()` per action only advances 2ms of real time, nowhere near enough for the joint's position servo to move anywhere close to the commanded target — I measured this directly: a full-strength action produced about 0.0002 rad of actual movement with only one substep. Holding the target across 50 substeps (100ms of simulated time per action) gives the servo real time to respond.

**Target sampling uses Gymnasium's seeded `self.np_random`, not the global NumPy RNG.** This wasn't the original implementation — I initially used plain `np.random.uniform()`, which meant `reset(seed=...)` silently didn't control target sampling at all, so two evaluation runs claiming "fixed seeds" weren't actually seeing the same targets. Fixed by routing all environment randomness through `self.np_random`, which is what `reset(seed=...)` actually controls. Worth calling out explicitly since it's exactly the kind of bug that looks fine until someone checks carefully.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Get the UR5e model
mkdir -p models && cd models
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
cd mujoco_menagerie && git sparse-checkout set universal_robots_ur5e && cd ../..
```

Quick smoke test — confirms the model loads, the env steps without errors, and passes Gymnasium's own API-contract checker:
```bash
python envs/reach_env.py
```

## Training

```bash
python training/train_ppo.py --timesteps 600000
python training/train_sac.py --timesteps 600000
python training/compare_results.py
```

Both algorithms train on the same 600k-timestep budget and a fixed `seed=42`, so the comparison below isn't confounded by one algorithm simply getting more data than the other.

`compare_results.py` runs 20 evaluation episodes for each algorithm, using `self.np_random`-seeded resets — meaning both algorithms are evaluated against the *same* 20 targets, not independently sampled ones.

## Results

| Algorithm | Success rate (20 episodes) | Mean final distance |
|-----------|------------------------------|----------------------|
| PPO       | 10%                           | 17.6 cm               |
| SAC       | 90%                           | 2.2 cm                 |

![Learning curves](results/learning_curves.png)

I ran this comparison twice. The first pass had two real problems: PPO trained on 300k timesteps against SAC's 600k, and the target-sampling seeding bug above meant the "20 fixed episodes" weren't actually matched between algorithms. Both issues are exactly the kind of thing that makes a comparison look controlled without actually being controlled. I fixed both, retrained PPO to match SAC's budget, and reran the whole evaluation. The result held: PPO landed at 10% again (17.6cm vs the original run's 16.4cm — consistent, not a fluke), while SAC's number shifted slightly (90% vs the original 95%, expected given it's now genuinely a different, previously-unseen set of 20 targets under the fixed seeding). The gap didn't close. That's a stronger result than the first pass, not just a repeated one — it survived a genuine attempt to break it.

SAC's advantage comes down to sample efficiency: its replay buffer lets it reuse every transition many times over, while PPO discards each batch after a single update. For a task like this — continuous control, dense reward, no strict on-policy requirement — that difference compounds significantly over training.

I checked this wasn't just a training-script artifact by deploying the SAC model through the actual ROS 2 bridge and watching it run live. Episode success rate and final distances in the live deployment closely matched the offline numbers above.

![Demo](results/demo.gif)

*(The red sphere is a visualization only — a coordinate the code tracks internally, not a physical object in the simulation.)*

## ROS 2 deployment

ROS 2 launches these nodes using your **system** Python, not the `venv` from Setup above, so the dependencies need to be available there too:

```bash
pip install -r requirements.txt --break-system-packages
```

Both nodes locate the `envs/` package via an environment variable — set once per terminal, or add to `~/.bashrc`:

```bash
export MUJOCO_ROS2_PROJECT_ROOT=/absolute/path/to/this/project
```

Then build and launch:

```bash
cd ros2_ws
colcon build --packages-select mujoco_arm_bridge
source install/setup.bash

ros2 launch mujoco_arm_bridge reach_demo.launch.py \
    model_path:=$MUJOCO_ROS2_PROJECT_ROOT/results/sac_reach_model.zip \
    algo:=sac
```

This starts two nodes:
- **`sim_node`** owns the MuJoCo simulation, publishes `/joint_states` (using the real UR5e joint names — `shoulder_pan_joint` through `wrist_3_joint`) and `/target_pose`, and subscribes to `/arm_joint_commands`.
- **`policy_node`** loads the trained model, maps incoming joint states by name (not array position, since a real driver's message ordering isn't guaranteed to match this project's internal convention), computes the wrist reference point via forward kinematics, and publishes joint commands.

**This is a simulation-to-topics bridge, not a certified drop-in real-hardware interface.** A real UR5e deployment via the official ROS 2 driver expects `ros2_control`-style commands (commonly a `FollowJointTrajectory` interface), not this project's normalized `[-1, 1]` RL action array over a custom `Float64MultiArray` topic. Getting from here to actual hardware would need a command-adapter layer between the policy's raw output and the robot's real controller — converting normalized deltas to real joint targets, enforcing velocity/acceleration limits, and handling the interface properly. The topic-based split between simulation and inference is the right architecture to build that adapter on top of; it isn't the adapter itself.

## Limitations / what I'd do next

- The wrist reference point (`wrist_3_link`) isn't the model's actual TCP (`attachment_site`, ~10cm offset) — see note above.
- Real UR5e ROS 2 deployment needs a command-adapter layer (see ROS 2 deployment section) — normalized RL actions aren't what a real driver's `ros2_control` interface expects.
- Training holds each action for 100ms of simulated time (50 physics substeps), but `sim_node`'s ROS timer runs at 50Hz (20ms wall-clock). That means each ROS timer tick currently advances 100ms of simulation — roughly 5x real-time — which works fine node-to-node in simulation, but isn't yet the same temporal semantics a real 50Hz control loop against physical hardware would have. Matching the decision interval to whatever the real control rate would be is a needed step before physical deployment, not just a detail.
- No camera or perception input yet — target position is given directly rather than detected. Natural next step given my other CV work: swap the ground-truth target for an OpenCV/YOLO-detected object pose.
- No collision avoidance. The reach task doesn't include obstacles; adding one with a collision-penalty term would be a reasonable extension.
- `policy_node` computes forward kinematics using its own MuJoCo model instance rather than a `tf2` lookup, which is the more standard ROS 2 approach. Functionally equivalent here, but a production deployment would likely use `tf2` for consistency with the rest of the ecosystem.
- Single training seed per algorithm (`seed=42`). The result is consistent across two independent evaluation runs under different confound conditions, which is reassuring, but multiple training seeds with reported variance would be the statistically rigorous version of this comparison.

## Requirements

- Python 3.10+
- MuJoCo 3.x (bundled Python bindings, no separate install needed)
- ROS 2 (Humble or newer) for the deployment portion — the RL training/comparison portion works standalone without ROS 2 installed