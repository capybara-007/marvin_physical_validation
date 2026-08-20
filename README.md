# Marvin Physical Validation

Marvin Pro 平行夹爪版本的独立 MuJoCo/Mink 轨迹回放与物理验证工具。

## 功能

- 双臂 7-DoF Mink IK/FK
- 左右平行夹爪控制
- UMI `session_ee` 文本轨迹回放
- 轨迹连续性、工作空间和碰撞评分
- MuJoCo 可视化

真实机器人控制尚未接入；本项目仅支持仿真验证。

## 环境安装

参考 `umi-vista`，使用 Conda 创建独立的 Python 3.10 环境：

```bash
cd /home/kernel/code/marvin_physical_validation
./setup_env.sh
conda activate marvin-physical-validation
```

`setup_env.sh` 会创建环境、安装依赖并自动运行模型冒烟测试。环境名可通过
`MARVIN_ENV_NAME` 修改：

```bash
MARVIN_ENV_NAME=my-marvin-env ./setup_env.sh
```

模型资源已经放在 `model/marvin_pro/`，无需额外下载。

## 模型冒烟测试

```bash
python3 smoke_test.py
```

正常输出应包含 `MODEL_OK`、`FK_OK`、`IK_OK` 和 `COLLISION_OK`。

## 回放轨迹

轨迹根目录应为 episode 层，例如：

```bash
./run_replay.sh \
  /home/kernel/code/umi-vista/physical_validation/converted_trajectories/episode_20260818_0009 \
  0
```

也可以直接运行：

```bash
export SCORE_FOLDER_PATH=/path/to/episode
python3 replay.py 0
```

默认搜索三级目录：`转换目录/50hz/session_ee_*`。如果 episode 中有多套转换结果，请先通过索引确认要回放的序号。

## 目录结构

```text
marvin_physical_validation/
├── configs/marvin_pro.json
├── model/marvin_pro/
├── robot/
├── utils/
├── replay.py
├── convert_episode_to_robot.py
├── smoke_test.py
├── setup_env.sh
├── run_replay.sh
└── requirements.txt
```

## 从原始 episode 直接可视化

`convert_episode_to_robot.py` 会读取原始 episode 中的：

```text
pose_data/pose_data_left.csv
pose_data/pose_data_right.csv
view_files/sensor_left_clamp_angle.json
view_files/sensor_right_clamp_angle.json
```

默认执行 AprilGrid → robot-base 坐标变换，并将当前 UGripper TCP 轨迹沿局部
`+Z` 方向反变换为 Marvin 控制的 gripper base。输出会缓存到本工程的
`converted_trajectories/`，因此可视化脚本可以直接接收原始 episode 根目录：

```bash
python test_full_vector_alignment.py \
  /home/kernel/Desktop/data/unified_episode/episode_20260818_0002 \
  --viewer
```

也可以单独执行转换：

```bash
python convert_episode_to_robot.py \
  /home/kernel/Desktop/data/unified_episode/episode_20260818_0002
```

如果希望保留输入 TCP 控制点而不做 `TCP → base` 偏移，指定：

```bash
python convert_episode_to_robot.py \
  /path/to/episode --control-point tcp
```
