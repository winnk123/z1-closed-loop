# Z1 远端部署

## 网络结构

Relay 和 NavDP worker 默认仅监听云端 loopback。Z1 主动建立本地端口转发：

```text
Z1 127.0.0.1:18888 -> SSH -> Cloud 127.0.0.1:18888
Relay 127.0.0.1:18888 -> NavDP 127.0.0.1:18889
```

云端防火墙只需允许 SSH，不应把 `18888`、`18889` 直接暴露到公网。

## 云端部署

```bash
git clone https://github.com/winnk123/z1-closed-loop.git
cd z1-closed-loop/cloud-relay
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

NavDP 不在本仓库内。把其源码与 checkpoint 部署在云端，并在 `.env` 设置绝对路径。

## 机器人部署

```bash
git clone https://github.com/winnk123/z1-closed-loop.git
cd z1-closed-loop/robot
python3 -m pip install -r requirements.txt
cp config/z1_planning.example.yaml config/z1_planning.yaml
```

为机器人创建只用于 tunnel 的 SSH key，并将公钥加入云端账号。私钥路径通过 `ULV_TUNNEL_KEY` 提供，不要提交到仓库。

## 更新

机器人、工作站和云端都以 GitHub `main` 为发布基线：

```bash
git pull --ff-only origin main
```

机器人端的现场修改先在独立分支提交并推送，再合并回 `main`，避免出现三份无法判断来源的代码。

不要把密码、私钥、API key、模型权重、采集数据或机器专用 `.env` 提交到 Git。
