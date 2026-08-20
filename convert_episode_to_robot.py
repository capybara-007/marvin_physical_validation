#!/usr/bin/env python3
"""Convert a raw dual-gripper episode into Marvin replay trajectories.

The raw episode contains pose CSV files in the AprilGrid frame and clamp-angle
JSON files.  This module synchronizes both hands at 50 Hz, maps AprilGrid axes
to the Marvin robot-base axes, optionally converts TCP poses back to gripper
base poses, and writes the ``50hz/session_ee_*`` layout consumed by replay.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


SAMPLE_PERIOD_SEC = 0.02
DEFAULT_TCP_TO_BASE_TZ = 0.129708200693
R_APRILGRID_TO_ROBOT = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=float
)
R_SLAM_WORLD_TO_ROBOT = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float
)


def _load_pose(path: Path) -> tuple[np.ndarray, np.ndarray, Rotation]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    time_sec = np.asarray(data["Timestamp_us"], dtype=float) * 1e-6
    position = np.column_stack([data["X"], data["Y"], data["Z"]])
    quaternion = np.column_stack(
        [data["Quat_X"], data["Quat_Y"], data["Quat_Z"], data["Quat_W"]]
    )
    return time_sec, position, Rotation.from_quat(quaternion)


def _interpolate_pose(
    source_time: np.ndarray,
    source_position: np.ndarray,
    source_rotation: Rotation,
    target_time: np.ndarray,
) -> tuple[np.ndarray, Rotation]:
    position = np.column_stack(
        [np.interp(target_time, source_time, source_position[:, axis]) for axis in range(3)]
    )
    return position, Slerp(source_time, source_rotation)(target_time)


def _load_gripper(path: Path, target_time: np.ndarray) -> np.ndarray:
    samples = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=float)
    time_sec = samples[:, 0] * 1e-9
    angle_rad = samples[:, 1]
    interpolated = np.interp(target_time, time_sec, angle_rad)
    return np.clip(-interpolated / 0.36 * 80.0, 0.0, 80.0)


def _relative_robot_pose(
    position: np.ndarray, rotation: Rotation, source_to_robot: np.ndarray
) -> np.ndarray:
    initial_position = position[0]
    initial_rotation_inverse = rotation[0].as_matrix().T
    relative_position = (source_to_robot @ (position - initial_position).T).T
    relative_rotation = rotation.as_matrix() @ initial_rotation_inverse
    relative_rotation = (
        source_to_robot[None, :, :]
        @ relative_rotation
        @ source_to_robot.T[None, :, :]
    )
    quaternion = Rotation.from_matrix(relative_rotation).as_quat()
    relative_position[0] = 0.0
    quaternion[0] = np.array([0.0, 0.0, 0.0, 1.0])
    return np.column_stack([relative_position, quaternion])


def _write_trajectory(path: Path, relative_time: np.ndarray, pose: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([relative_time, pose]), fmt="%.9f")


def _write_gripper(path: Path, relative_time: np.ndarray, gripper: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([relative_time, gripper]), fmt="%.9f")


def output_path_for_episode(
    episode: Path, output_root: Path, source_frame: str, control_point: str
) -> Path:
    suffix = "_base_to_robot" if control_point == "base" else "_to_robot"
    frame_name = "aprilgrid" if source_frame == "aprilgrid" else "slam_world"
    return output_root / episode.name / f"{frame_name}{suffix}"


def convert_episode(
    episode: Path,
    output_root: Path,
    *,
    source_frame: str = "aprilgrid",
    control_point: str = "base",
    tcp_to_base_tz: float = DEFAULT_TCP_TO_BASE_TZ,
    overwrite: bool = False,
) -> Path:
    """Convert one raw episode and return its replay trajectory directory.

    The raw CSVs produced by the current UGripper pipeline are TCP poses.  The
    default therefore converts them to gripper-base control points, matching
    Marvin's ``left_tool_site`` and ``right_tool_site`` semantics.  Pass
    ``control_point='tcp'`` when the input should remain TCP.
    """
    episode = episode.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if source_frame not in ("aprilgrid", "slam_world"):
        raise ValueError(f"unsupported source frame: {source_frame!r}")
    if control_point not in ("base", "tcp"):
        raise ValueError(f"unsupported control point: {control_point!r}")
    pose_dir = episode / "pose_data"
    pose_paths = {side: pose_dir / f"pose_data_{side}.csv" for side in ("left", "right")}
    gripper_paths = {
        side: episode / "view_files" / f"sensor_{side}_clamp_angle.json"
        for side in ("left", "right")
    }
    for path in [*pose_paths.values(), *gripper_paths.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)

    result_root = output_path_for_episode(episode, output_root, source_frame, control_point)
    metadata_path = result_root / "conversion_metadata.json"
    if metadata_path.is_file() and not overwrite:
        return result_root

    source_to_robot = (
        R_SLAM_WORLD_TO_ROBOT if source_frame == "slam_world" else R_APRILGRID_TO_ROBOT
    )
    poses = {side: _load_pose(path) for side, path in pose_paths.items()}
    if control_point == "base":
        if tcp_to_base_tz <= 0.0:
            raise ValueError("tcp_to_base_tz must be positive for base conversion")
        poses = {
            side: (
                time_sec,
                position - rotation.apply(np.broadcast_to([0.0, 0.0, tcp_to_base_tz], position.shape)),
                rotation,
            )
            for side, (time_sec, position, rotation) in poses.items()
        }

    start_time = max(value[0][0] for value in poses.values())
    stop_time = min(value[0][-1] for value in poses.values())
    target_time = np.arange(start_time, stop_time, SAMPLE_PERIOD_SEC)
    if target_time.size < 2:
        raise ValueError("the synchronized left/right time range is too short")
    relative_time = target_time - target_time[0]

    interpolated: dict[str, tuple[np.ndarray, Rotation]] = {}
    grippers: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        source_time, source_position, source_rotation = poses[side]
        interpolated[side] = _interpolate_pose(
            source_time, source_position, source_rotation, target_time
        )
        grippers[side] = _load_gripper(gripper_paths[side], target_time)

    initial_positions = {
        side: interpolated[side][0][0].copy() for side in ("left", "right")
    }
    initial_vector_robot = (
        source_to_robot @ (initial_positions["left"] - initial_positions["right"])
    )
    stats: dict[str, object] = {}
    for side in ("left", "right"):
        source_time, _, _ = poses[side]
        position, rotation = interpolated[side]
        pose = _relative_robot_pose(position, rotation, source_to_robot)
        session_suffix = episode.name.rsplit("_", maxsplit=1)[-1]
        session_root = result_root / "50hz" / f"session_ee_{session_suffix}"
        _write_trajectory(
            session_root / side / "Merged_Trajectory" / "merged_trajectory.txt",
            relative_time,
            pose,
        )
        _write_gripper(
            session_root / side / "Clamp_Data" / "clamp_data_tum.txt",
            relative_time,
            grippers[side],
        )
        stats[side] = {
            "source_initial_index": int(np.searchsorted(source_time, start_time)),
            "first_output": pose[0].tolist() + [float(grippers[side][0])],
            "position_min_robot_m": pose[:, :3].min(axis=0).tolist(),
            "position_max_robot_m": pose[:, :3].max(axis=0).tolist(),
            "quaternion_norm_min": float(np.linalg.norm(pose[:, 3:7], axis=1).min()),
            "quaternion_norm_max": float(np.linalg.norm(pose[:, 3:7], axis=1).max()),
            "gripper_output_min": float(grippers[side].min()),
            "gripper_output_max": float(grippers[side].max()),
        }

    metadata = {
        "source": str(episode),
        "source_pose": (
            "T_source_tcp transformed by inverse(T_base_tcp)"
            if control_point == "base"
            else "T_source_tcp"
        ),
        "control_point": control_point,
        "tcp_to_base_tz_m": float(tcp_to_base_tz) if control_point == "base" else None,
        "initial_control_point_vector_robot_m": initial_vector_robot.tolist(),
        "initial_control_point_distance_m": float(np.linalg.norm(initial_vector_robot)),
        # Compatibility with the current replay helper; this is the distance
        # of the actual control points, not necessarily a physical TCP distance.
        "initial_tcp_distance_m": float(np.linalg.norm(initial_vector_robot)),
        "sample_rate_hz": 1.0 / SAMPLE_PERIOD_SEC,
        "frame_count": int(target_time.size),
        "duration_sec": float(relative_time[-1]),
        "source_frame": source_frame,
        "R_source_to_robot": source_to_robot.tolist(),
        "det_R_source_to_robot": float(np.linalg.det(source_to_robot)),
        "axis_mapping": (
            {"robot_x": "-slam_world_y", "robot_y": "slam_world_x", "robot_z": "slam_world_z"}
            if source_frame == "slam_world"
            else {"robot_x": "grid_z", "robot_y": "-grid_x", "robot_z": "-grid_y"}
        ),
        "translation_formula": "dp_robot = R_source_to_robot @ (p_now - p_initial)",
        "rotation_formula": "dR_robot = R_source_to_robot @ (R_now @ R_initial.T) @ R_source_to_robot.T",
        "gripper_mapping": "clip(-angle_rad / 0.36 * 80, 0, 80)",
        "stats": stats,
    }
    result_root.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result_root


def resolve_trajectory_input(
    input_path: Path,
    *,
    output_root: Path | None = None,
    source_frame: str = "aprilgrid",
    control_point: str = "base",
    tcp_to_base_tz: float = DEFAULT_TCP_TO_BASE_TZ,
) -> Path:
    """Return a replay directory, converting a raw episode when necessary."""
    input_path = input_path.expanduser().resolve()
    if (input_path / "50hz").is_dir():
        return input_path
    if not (input_path / "pose_data").is_dir():
        raise FileNotFoundError(
            f"{input_path} is neither a converted trajectory directory (missing 50hz/) "
            "nor a raw episode directory (missing pose_data/)"
        )
    if output_root is None:
        output_root = Path(__file__).resolve().parent / "converted_trajectories"
    result_root = output_path_for_episode(input_path, output_root, source_frame, control_point)
    if (result_root / "conversion_metadata.json").is_file() and (result_root / "50hz").is_dir():
        return result_root
    return convert_episode(
        input_path,
        output_root,
        source_frame=source_frame,
        control_point=control_point,
        tcp_to_base_tz=tcp_to_base_tz,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="raw episode root directory")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "converted_trajectories",
    )
    parser.add_argument("--source-frame", choices=("aprilgrid", "slam_world"), default="aprilgrid")
    parser.add_argument(
        "--control-point", choices=("base", "tcp"), default="base",
        help="output control point; base is Marvin's default control point",
    )
    parser.add_argument("--tcp-to-base-tz", type=float, default=DEFAULT_TCP_TO_BASE_TZ)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(convert_episode(
        args.episode,
        args.output_root,
        source_frame=args.source_frame,
        control_point=args.control_point,
        tcp_to_base_tz=args.tcp_to_base_tz,
        overwrite=args.overwrite,
    ))


if __name__ == "__main__":
    main()
