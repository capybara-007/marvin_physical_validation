#!/usr/bin/env python3
import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["ROBOT_NAME"] = "marvin_pro"

from robot.marvin_pro_mink import MarvinProMink


def main() -> None:
    robot = MarvinProMink()
    hand_poses = robot.solve_fk()
    qpos = robot.solve_ik(hand_poses)
    feedback = robot.get_sim_feedback()
    collision = robot.check_collision_pairs()

    assert robot.model.nq == 18
    assert len(hand_poses) == 2
    assert all(pose.shape == (7,) for pose in hand_poses)
    assert qpos.shape == (16,)
    assert np.all(np.isfinite(qpos))
    assert len(feedback["qpos"]["left_arm"]) == 7
    assert len(feedback["qpos"]["right_arm"]) == 7

    print("MODEL_OK", robot.model.nq, robot.model.njnt, robot.model.nsite)
    print("FK_OK", [pose.tolist() for pose in hand_poses])
    print("IK_OK", qpos.tolist())
    print("COLLISION_OK", collision)


if __name__ == "__main__":
    main()
