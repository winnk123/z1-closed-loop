# Cloud Relay

Relay 提供 LA、VA 和 NavDP HTTP 接口。NavDP 源码与 checkpoint 属于外部依赖，不提交到本仓库。

## Setup

```bash
cd cloud-relay
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

将 `.env` 中的路径和 API key 替换为本机值，然后加载：

```bash
set -a
. ./.env
set +a
```

## Run

终端 1 启动 NavDP worker：

```bash
./run_navdp_worker.sh
```

终端 2 启动 Relay：

```bash
./run_relay.sh
```

健康检查：

```bash
curl http://127.0.0.1:18888/health
```

应看到 `ok: true` 和 `navdp_ready: true`。生产使用建议保持 Relay 监听 `127.0.0.1`，由机器人通过 SSH tunnel 访问。
