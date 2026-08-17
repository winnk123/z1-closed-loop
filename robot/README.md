# Robot

## Prerequisites

- MagicBot Z1 系统环境，默认 `/opt/eame/setup.bash`
- ROS 2 Humble 和 `motion_msgs`
- MagicBot Z1 SDK Python binding
- Intel RealSense D435 或 Sensor Manager RGB-D topics
- Python 3.10+

## Setup

```bash
cd robot
python3 -m pip install -r requirements.txt
cp config/z1_planning.example.yaml config/z1_planning.yaml
cp .z1_env.example .z1_env
```

编辑 `config/z1_planning.yaml` 中的相机序列号、任务和规划参数，并在 `.z1_env` 中填写 SDK、ROS 路径和本机 IP。`.z1_env` 不会提交 Git。也可直接使用环境变量：

```bash
export EAME_SETUP=/opt/eame/setup.bash
export Z1_SDK_ROOT=/path/to/magicbot-z1_sdk-main
export MOTION_MSGS_PREFIX=/opt/eame/motion_msgs
export Z1_LOCAL_IP=<robot-network-interface-ip>
```

如果 Relay 位于云端，先配置并启动隧道：

```bash
export ULV_SERVER_HOST=<relay-server-host>
export ULV_SERVER_USER=<relay-server-user>
export ULV_TUNNEL_KEY=$HOME/.ssh/ulv_relay_ed25519
./ops/start_tunnel.sh
```

## Run

先运行只读检查：

```bash
./ops/check_z1_closed_loop_readiness.sh
```

确认机器人处于安全测试区并有急停操作员后启动：

```bash
./scripts/run_baseline_stop_remote_navdp.sh
```

可用第一个参数覆盖网卡 IP，也可用环境变量 `Z1_LOCAL_IP`。任务文本可通过 `Z1_INSTRUCTION` 和 `Z1_TARGET_DESCRIPTION` 覆盖。

使机器人进入 balance stand：

```bash
./ops/start_z1_balance_stand.sh
```

紧急停止辅助命令：

```bash
./ops/send_z1_zero_joystick.sh
```
