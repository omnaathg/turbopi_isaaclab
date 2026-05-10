"""Runtime helper for ACT chunked inference."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from . import DEFAULT_ACTION_DIM, DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH
from .model import load_checkpoint


@dataclass
class ACTRuntimeConfig:
    vx_cap: float = 0.45
    vy_cap: float = 0.35
    wz_cap: float = 2.0
    smoothing: float = 0.35
    replan_every: int = 5


class ACTPolicyRuntime:
    def __init__(self, checkpoint: str | Path, *, task: str, device: str = "auto", runtime_cfg: ACTRuntimeConfig | None = None):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model, payload = load_checkpoint(Path(checkpoint), map_location=self.device)
        self.model.to(self.device).eval()
        extra = payload.get("extra", {})
        self.task_names = list(extra.get("task_names", ["go_left", "go_right"]))
        self.task_to_index = {name: idx for idx, name in enumerate(self.task_names)}
        self.cfg = runtime_cfg or ACTRuntimeConfig()
        self.queue: deque[np.ndarray] = deque()
        self.smoothed = np.zeros(DEFAULT_ACTION_DIM, dtype=np.float32)
        self.set_task(task)

    @property
    def image_width(self) -> int:
        return int(self.model.config.image_width)

    @property
    def image_height(self) -> int:
        return int(self.model.config.image_height)

    def set_task(self, task: str) -> None:
        if task not in self.task_to_index:
            raise ValueError(f"Unknown task '{task}'. Available: {self.task_names}")
        self.task = task
        self.task_index = self.task_to_index[task]
        self.queue.clear()
        self.smoothed[:] = 0.0

    def preprocess(self, image_rgb: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(image_rgb)
        w, h = self.image_width, self.image_height
        if image.size != (w, h):
            image = image.resize((w, h), Image.Resampling.BILINEAR)
        return TF.to_tensor(image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict_chunk(self, image_rgb: np.ndarray) -> np.ndarray:
        image = self.preprocess(image_rgb)
        task_id = torch.tensor([self.task_index], dtype=torch.long, device=self.device)
        output = self.model(image, task_id, action_chunk=None)["action"][0].detach().cpu().numpy().astype(np.float32)
        return output

    def predict(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.queue:
            chunk = self.predict_chunk(image_rgb)
            self.queue.extend(chunk)
        raw = np.asarray(self.queue.popleft(), dtype=np.float32)
        alpha = float(np.clip(self.cfg.smoothing, 0.0, 1.0))
        self.smoothed = (1.0 - alpha) * self.smoothed + alpha * raw
        caps = np.asarray([self.cfg.vx_cap, self.cfg.vy_cap, self.cfg.wz_cap, 1.0], dtype=np.float32)
        command = np.clip(self.smoothed, -1.0, 1.0) * caps
        return raw, command
