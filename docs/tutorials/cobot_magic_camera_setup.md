# RealSense D435 配置

## 识别相机

在机器人上执行：

```bash
rs-enumerate-devices
```

记录目标 D435 的 serial，并设置：

```bash
export Z1_CAMERA_SERIAL=<realsense-serial>
```

同一个值也要写入 `robot/config/z1_planning.yaml` 的 `camera.serial`。

## 默认流配置

- RGB: `1280x720 @ 15 FPS`
- Depth: `640x480 @ 15 FPS`
- Depth 对齐到 color
- Color: JPEG
- Depth: PNG `16UC1`，单位毫米

USB 带宽不足时先确认相机连接在 USB 3.x。不要通过降低深度精度来掩盖链路问题。

## Sensor Manager bridge

```bash
cd z1-closed-loop/robot
export Z1_CAMERA_SERIAL=<realsense-serial>
./ops/start_sensor_manager.sh
```

bridge 发布：

- `/sensor/rgbd_image`
- `/sensor/rgbd_depth_image`
- `/sensor/rgbd_camera_info`

检查 topic：

```bash
ros2 topic hz /sensor/rgbd_image
ros2 topic hz /sensor/rgbd_depth_image
ros2 topic echo --once /sensor/rgbd_camera_info
```

如果 D435 断开后重连，bridge 会重新创建 RealSense pipeline。闭环运行前仍需重新执行 readiness 检查。

## 常见问题

| 现象 | 检查项 |
|---|---|
| 找不到设备 | serial 是否正确、USB 是否枚举 |
| 帧超时 | USB 3.x 带宽、电源、是否有第二个进程占用相机 |
| RGB 与深度不匹配 | 必须使用同步帧并将 depth 对齐到 color |
| topic 有数据但 readiness 失败 | 检查 camera info、时间戳和压缩格式 |
