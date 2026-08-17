# Z1 闭环运行教程

本文对应本仓库的 remote NavDP 主链。

## 1. 启动顺序

```text
Cloud: NavDP worker -> Relay
Robot: Sensor Manager -> SSH tunnel -> readiness -> closed-loop runner
```

## 2. 云端

```bash
cd z1-closed-loop/cloud-relay
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，至少填写 `DASHSCOPE_API_KEY`、`NAVDP_ROOT` 和 `NAVDP_CHECKPOINT`，然后在两个终端分别运行：

```bash
set -a; . ./.env; set +a
./run_navdp_worker.sh
```

```bash
set -a; . ./.env; set +a
./run_relay.sh
```

验证：

```bash
curl -fsS http://127.0.0.1:18888/health
```

结果必须包含 `"ok": true` 和 `"navdp_ready": true`。

## 3. 机器人端

```bash
cd z1-closed-loop/robot
python3 -m pip install -r requirements.txt
cp config/z1_planning.example.yaml config/z1_planning.yaml
```

编辑配置中的 `camera.serial`、任务文本和控制参数。设置机器人环境：

```bash
export EAME_SETUP=/opt/eame/setup.bash
export Z1_SDK_ROOT=/absolute/path/to/magicbot-z1_sdk-main
export MOTION_MSGS_PREFIX=/opt/eame/motion_msgs
export Z1_LOCAL_IP=<robot-network-interface-ip>
export Z1_CAMERA_SERIAL=<realsense-serial>
```

如需启动兼容 Sensor Manager 的 RealSense bridge：

```bash
./ops/start_sensor_manager.sh
```

建立到 Relay 的 SSH tunnel：

```bash
export ULV_SERVER_HOST=<relay-server-host>
export ULV_SERVER_USER=<relay-server-user>
export ULV_TUNNEL_KEY=$HOME/.ssh/ulv_relay_ed25519
./ops/start_tunnel.sh
```

## 4. 运行前检查

```bash
./ops/check_z1_closed_loop_readiness.sh
```

只有 `runtime`、`camera` 和 `relay` 都返回 `ready: true` 才能运行。该检查不创建 SDK sender，也不会发运动命令。

## 5. 启动闭环

```bash
export Z1_INSTRUCTION='找到桌上的水杯并且停下。'
export Z1_TARGET_DESCRIPTION='桌上的水杯。'
./scripts/run_baseline_stop_remote_navdp.sh
```

主脚本会显式传入三个运动门控参数：`--execute`、`--enable-motion` 和确认字符串。运行记录写入 `captures/missions/lavira/`，该目录不会提交 Git。

紧急时由现场操作员使用硬件急停。软件零速辅助命令为：

```bash
./ops/send_z1_zero_joystick.sh
```

## 6. 循环原理

1. LA 根据候选视角决定战略朝向，或返回 `STOP`。
2. 机器人原地转向，等待姿态稳定后采集同步 RGB-D。
3. VA 在当前 RGB 图像中给出目标框。
4. 机器人端用深度和相机内参将目标框反投影为局部 point goal。
5. Relay 将 RGB-D、内参和 point goal 转发给 NavDP worker。
6. NavDP 返回候选局部轨迹，机器人端安全桥限速后跟踪。
7. 执行中周期重规划；到达安全截断终点后返回 LA，直到停止或触发安全条件。
