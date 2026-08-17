# Z1 Closed Loop

MagicBot Z1 的 LA / VA / NavDP 视觉闭环运行仓库。仓库只保留当前 remote NavDP 主链，不包含测试、benchmark、历史产物和模型权重。

```text
z1-closed-loop/
├── robot/                 # Z1 相机、规划、安全控制与主入口
├── cloud-relay/           # DashScope LA/VA 与 NavDP HTTP Relay
└── docs/tutorials/        # 部署和操作教程
```

## Pipeline

```text
任务指令
  -> LA 使用候选视角图像决定战略方向或 STOP
  -> Z1 原地转向并采集同步 RGB-D
  -> VA 在当前图像中定位目标
  -> 深度反投影得到机器人坐标系局部目标
  -> Cloud Relay 转发 RGB-D 和目标到 NavDP worker
  -> NavDP 返回局部轨迹
  -> Z1 安全桥限制速度并跟踪轨迹
  -> 周期重规划，直到 LA 返回 STOP、超时或安全状态触发
```

机器人主动建立 SSH local-forward，Relay 只监听云端 `127.0.0.1:18888`，无需开放公网推理端口。

## Quick Start

1. 按 [Cloud Relay](cloud-relay/README.md) 启动 NavDP worker 和 Relay。
2. 按 [Robot](robot/README.md) 配置 Z1、相机和 SSH tunnel。
3. 先执行只读 readiness 检查，通过后再启动运动闭环。

详细步骤见 [闭环教程](docs/tutorials/z1_closed_loop_guide.md) 和 [远端部署](docs/tutorials/z1_remote_deployment.md)。

## Safety

真实机器人运行必须预留急停操作员和无障碍测试区。主入口默认不会发运动指令，只有同时传入 `--execute`、`--enable-motion` 和确认字符串时才会创建运动发送器。
