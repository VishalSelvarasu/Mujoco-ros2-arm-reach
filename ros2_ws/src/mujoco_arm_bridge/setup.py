from setuptools import find_packages, setup
import os
from glob import glob

package_name = "mujoco_arm_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Vishal Selvarasu",
    maintainer_email="82762526+VishalSelvarasu@users.noreply.github.com",
    description="ROS 2 bridge for deploying a MuJoCo-trained UR5e reach policy",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "policy_node = mujoco_arm_bridge.policy_node:main",
            "sim_node = mujoco_arm_bridge.sim_node:main",
        ],
    },
)