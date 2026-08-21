# Marvin Physical Validation

Marvin Pro 平行夹爪版本的独立 MuJoCo/Mink 轨迹回放与物理验证工具。

## 功能

- 双臂 7-DoF Mink IK/FK
- 左右平行夹爪控制
- UMI `session_ee` 文本轨迹回放
- 戴盟 UGripper base 轨迹到 TCP 的位姿转换
- 以 Marvin 双指末端中心作为 IK 控制点
- 轨迹连续性、工作空间和碰撞评分
- MuJoCo 可视化

真实机器人控制尚未接入；本项目仅支持仿真验证。

## 环境安装

使用 Conda 创建独立的 Python 3.10 环境：

```bash
cd /home/kernel/code/marvin_physical_validation
./setup_env.sh
conda activate marvin-physical-validation
```

`setup_env.sh` 会创建环境并安装依赖。环境名可通过
`MARVIN_ENV_NAME` 修改：

```bash
MARVIN_ENV_NAME=my-marvin-env ./setup_env.sh
```

模型资源已经放在 `model/marvin_pro/`，无需额外下载。

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

原始 CSV 中的位姿是戴盟 UGripper base 位姿（戴盟文档里的TCP不是夹指中点，而是base）。转换器会先使用代码中内置的
`T_UGRIPPER_BASE_TCP` 计算戴盟 TCP 位姿：

```text
T_source_tcp = T_source_base @ T_base_tcp

T_base_tcp =
[[1, 0, 0, 0               ],
 [0, 1, 0, 0               ],
 [0, 0, 1, 0.129708200693  ],
 [0, 0, 0, 1               ]]
```

也就是 TCP 相对戴盟 base 沿局部 `+Z` 方向偏移
`0.129708200693 m`，两者坐标轴方向一致。该矩阵直接定义在`convert_episode_to_robot.py` 
得到戴盟 TCP 位姿后，转换器再执行 AprilGrid → Marvin robot-base
坐标变换，并生成以首帧为零点的相对 TCP 轨迹。MuJoCo 模型中的
`left_tool_site` 和 `right_tool_site` 位于 Marvin 两个夹指末端的中心，
因此回放时的完整控制链为：

```text
戴盟 gripper base 位姿
  → 戴盟 TCP 位姿
  → Marvin 双指末端中心目标位姿
  → Mink IK
```

Marvin 双指末端中心相对其夹爪 base 的位置约为：

```text
[0.000795, -0.002959, 0.162710] m
```

输出会缓存到本工程的 `converted_trajectories/`。转换元数据包含
`conversion_format_version`、`control_point: tcp` 和实际使用的
`T_base_tcp`；旧的 base 控制点缓存或矩阵不一致的缓存会自动失效并重新生成。
因此可视化脚本可以直接接收原始 episode 根目录：

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

如需无条件重写已有转换结果，可使用：

```bash
python convert_episode_to_robot.py \
  /home/kernel/Desktop/data/unified_episode/episode_20260818_0002 \
  --overwrite
```
