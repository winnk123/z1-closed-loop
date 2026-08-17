"""
In-process iPlanner adapter for ULV relay server.

Wraps uni-lavira's IPlannerAgent, providing GPU-accelerated trajectory planning.
Based on cobot_magic/robot/iplanner_client.py.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_IPLANNER_DIR = _HERE / "iplanner"
if str(_IPLANNER_DIR) not in sys.path:
    sys.path.insert(0, str(_IPLANNER_DIR))


class IPlannerAdapter:
    """In-process wrapper around iPlanner IPlannerAgent.

    Handles depth resize (224x224), tensor conversion, and trajectory extraction.
    """

    TARGET_SIZE: Tuple[int, int] = (224, 224)

    # Default intrinsics (Orbbec Gemini 336L-alike, 640x480).
    # fx, fy, cx, cy
    DEFAULT_INTRINSICS = [611.0, 0.0, 320.0, 0.0, 611.0, 240.0, 0.0, 0.0, 1.0]

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        intrinsics: Optional[List[float]] = None,
        device: str = "cuda:0",
    ) -> None:
        import torch
        from iplanner.iplanner_agent import IPlannerAgent

        self._torch = torch
        self.device = device

        _intrinsics = intrinsics or self.DEFAULT_INTRINSICS

        print(f"[iPlanner] Loading checkpoint={checkpoint_path} device={device}")
        self.agent = IPlannerAgent(
            image_intrinsic=torch.tensor(_intrinsics),
            model_path=checkpoint_path,
            model_config_path=config_path,
            device=device,
        )
        print("[iPlanner] Ready")

    def plan(
        self,
        depth_image: np.ndarray,
        local_goal_xy: Tuple[float, float],
    ) -> Optional[List[List[float]]]:
        """Plan a robot-frame 2D trajectory toward local_goal_xy.

        Args:
            depth_image: Front-camera depth image (metres), HxW float32.
            local_goal_xy: Goal (x, y) in robot frame (x forward, y left).

        Returns:
            List of [x, y] robot-frame waypoints, or None on failure.
        """
        import cv2

        torch = self._torch
        try:
            local_goal_x, local_goal_y = float(local_goal_xy[0]), float(local_goal_xy[1])

            depth_resized = cv2.resize(
                depth_image, self.TARGET_SIZE, interpolation=cv2.INTER_NEAREST
            )
            depth_input = depth_resized.astype(np.float32)[np.newaxis, :, :, np.newaxis]
            depth_tensor = torch.as_tensor(depth_input, device=self.agent.device)

            goal_input = np.array(
                [[local_goal_x, local_goal_y, 0.0]], dtype=np.float32
            )
            goal_tensor = torch.as_tensor(goal_input, device=self.agent.device)

            with torch.no_grad():
                _, trajectory, _ = self.agent.step_pointgoal(depth_tensor, goal_tensor)

            traj_list = trajectory.cpu().numpy().tolist()
            if not traj_list:
                return None

            raw_path = np.array(traj_list[0])
            if raw_path.ndim == 2 and raw_path.shape[1] >= 2:
                return raw_path.tolist()
            return None
        except Exception:
            print(f"[iPlanner] Error: {traceback.format_exc()}")
            return None
