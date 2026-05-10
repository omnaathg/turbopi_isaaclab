"""Language-conditioned ACT + CVAE policy package for TurboPi mountain cliff tasks."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

DEFAULT_ACT_DATA_ROOT = str(REPO_ROOT / "data" / "act_mountain_cliff")
DEFAULT_IMAGE_WIDTH = 64
DEFAULT_IMAGE_HEIGHT = 64
DEFAULT_ACTION_CHUNK_SIZE = 8
DEFAULT_ACTION_DIM = 4
DEFAULT_TASKS = ("go_left", "go_right")
EXPECTED_ACT_CVAE_PARAM_COUNT = 0  # 0 = auto (no fixed-count calibration)
