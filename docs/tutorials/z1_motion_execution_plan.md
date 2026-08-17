# Z1 运动执行与安全约束

## 控制链

```text
NavDP trajectory
  -> rolling replan terminal check
  -> path follower
  -> forward/yaw velocity limits
  -> runtime safety bridge
  -> MagicBot high-level joystick command
```

一个战略子目标的执行顺序是：LA 选定视角与方向，机器人完成 yaw scan 并稳定；VA 在当前 RGB-D 帧中检测目标；深度反投影得到 point goal；NavDP 返回轨迹；首次轨迹只截去一次 `0.50 m` 安全余量，得到固定的 odom 终点。后续 NavDP 重规划必须回到这个固定终点，终点误差超过 `0.35 m` 的轨迹会被拒绝。机器人进入固定终点 `1.00 m` 容差后回到 LA，而不是每次重规划都缩短终点。

## 周期与频率

| 环节 | 当前值 | 说明 |
|---|---:|---|
| 直行路径跟踪配置 | `0.05 s`，目标 20 Hz | `PathFollowerConfig.control_period_s` 的设计值。当前直行主循环没有固定 sleep，实际发送频率由 ROS 回调、相机和规划处理耗时决定，不能把 20 Hz 当作强制上限或下限。 |
| yaw 转向控制 | `0.05 s`，20 Hz | `drive_to_yaw()` 每轮显式 sleep `0.05 s`。 |
| 默认重规划间隔 | `0.40 s` | `run_z1_lavira_closed_loop.py` 的 CLI 默认值。 |
| 发布启动脚本重规划间隔 | `2.0 s` | `run_baseline_stop_remote_navdp.sh` 显式覆盖默认值，当前实机运行使用它。 |
| 最大计划年龄 | `1.00 s` | 超过此时间的 NavDP 结果不进入 follower。 |
| 局部前视距离 | `0.30 m` | 视觉闭环 admission 的配置值；Baseline 模式对首次路径保留固定安全终点。 |
| D435 数据流 | RGB `1280x720@15`，depth `640x480@15` | 需要同步 RGB-D 和相机内参。 |

最近一次实机闭环审计记录了 `26,925` 条控制样本。这个数字只反映该次任务的实际 loop 节奏，不应替代对控制频率的正式约束。若要将直行控制固定为 20 Hz，需要在 `execute_subgoal()` 中加入节拍调度，而不是只修改 `control_period_s`。

## 跟踪与速度限制

路径跟踪使用 lookahead `0.45 m`，巡航速度和最大前进速度均为 `0.10 m/s`。横向偏差超过 `0.75 m`、航向误差超过 `90 deg`、单个局部路径运行超过 `60 s`，或 odom 超过 `0.30 s` 未更新时，follower 会停止并返回失败。

Safety Bridge 对每一条 follower 输出再次限幅和限加速度：

| 项目 | 限制 |
|---|---:|
| 前进速度 | `0.10 m/s` |
| 倒车 | 禁止，`0.0 m/s` |
| yaw 角速度 | `0.20 rad/s` |
| 前进加速度 | `0.10 m/s^2` |
| yaw 加速度 | `0.20 rad/s^2` |
| joystick axis 绝对值 | `0.40` |
| axis 变化率 | `0.40 /s` |

当前标定使用 `forward_mps_per_axis = 1/3`、`yaw_rps_per_axis = 0.50`。也就是说，速度命令先被夹到上述物理量范围，再映射为高层 joystick axis；yaw 轴符号会按 Z1 SDK 坐标约定反向。以上标定必须在每台机器人重新确认。

## 看门狗与停止

Bridge 每次发送前要求：运动显式启用、命令时间戳不超过 `0.20 s`、IMU 不超过 `0.20 s`、mission odom 不超过 `0.50 s`、机器人未悬空、未急停且 FSM 为 `46`（balance stand）。任意一项失败都发送零命令并锁存停止原因。任务结束、异常或 sender 关闭时还会额外发送零 joystick 命令。

`run_z1_lavira_closed_loop.py` 默认只审计。创建真实运动 sender 必须同时满足：

- `--execute`
- `--enable-motion`
- `--confirm=EXECUTE_Z1_LAVIRA_CLOSED_LOOP`

运行脚本 `run_baseline_stop_remote_navdp.sh` 已显式提供这些参数，因此执行前必须人工完成 readiness 和现场检查。

## 现场检查

1. 清空机器人周围测试区域，确认地面平整。
2. 确认硬件急停可用，并安排独立操作员负责急停。
3. 确认机器人未悬空、没有 estop、FSM 为 balance stand。
4. 确认 odometry、IMU、RGB-D 和 Relay 数据新鲜。
5. 先使用低速度、小距离和单一目标验证。

## 自动停止条件

- LA 返回 `STOP`
- 达到最大任务步数或总运行时长
- odometry / IMU 超时
- estop、悬空或 FSM 异常
- 路径过期、重规划失败或终点误差超限
- Relay、相机或 NavDP 请求失败

单个战略子目标连续两次规划失败也会终止该子目标。LA 或 VA 请求停止时，系统只允许使用当前检测结果做一次最终接近；没有有效 bbox 则直接结束。

软件停止不能替代硬件急停。`ops/send_z1_zero_joystick.sh` 只发送零速 joystick command，用于辅助收尾。
