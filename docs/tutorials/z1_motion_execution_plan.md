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

软件停止不能替代硬件急停。`ops/send_z1_zero_joystick.sh` 只发送零速 joystick command，用于辅助收尾。
