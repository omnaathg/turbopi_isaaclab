"""Dataset utilities for ACT + language + CVAE training."""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from . import DEFAULT_ACTION_CHUNK_SIZE, DEFAULT_ACTION_DIM, DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH

try:
    import av
except ImportError:  # pragma: no cover
    av = None


@dataclass(frozen=True)
class EpisodeRecord:
    episode_dir: Path
    session_name: str
    num_frames: int
    task: str
    task_index_hint: int | None = None


@dataclass(frozen=True)
class SampleIndex:
    episode_idx: int
    frame_idx: int


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def discover_session_dirs(episodes_root: Path | str) -> list[Path]:
    root = Path(episodes_root)
    if not root.exists():
        return []
    return [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and any(child.is_dir() and child.name.startswith("episode_") for child in path.iterdir())
    ]


def _session_tasks(session_dir: Path) -> list[str]:
    tasks_path = session_dir / "tasks.json"
    if tasks_path.exists():
        try:
            data = json.loads(tasks_path.read_text(encoding="utf-8"))
        except Exception:
            data = None
        if isinstance(data, list):
            return [str(item) for item in data]
    mapping = _read_json(session_dir / "task_mapping.json")
    tasks = mapping.get("tasks")
    return [str(item) for item in tasks] if isinstance(tasks, list) else []


def _is_act_episode(session_info: dict, episode_info: dict) -> bool:
    family = episode_info.get("mode_family") or session_info.get("mode_family")
    policy = episode_info.get("policy_family") or session_info.get("policy_family")
    if str(policy) == "act_cvae" or str(family) == "act":
        return True
    return False


def discover_episodes(episodes_dir: Path | str) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for episode_dir in sorted(Path(episodes_dir).glob("**/episode_*")):
        parquet_path = episode_dir / "data.parquet"
        video_path = episode_dir / "video.mp4"
        info_path = episode_dir / "episode_info.json"
        if not episode_dir.is_dir() or not parquet_path.exists() or not video_path.exists() or not info_path.exists():
            continue
        info = _read_json(info_path)
        session_info = _read_json(episode_dir.parent / "session_info.json")
        if not _is_act_episode(session_info, info):
            continue
        df = pd.read_parquet(parquet_path, columns=["task", "task_index"])
        if df.empty:
            continue
        task = str(df["task"].iloc[0])
        try:
            task_index_hint = int(df["task_index"].iloc[0])
        except Exception:
            task_index_hint = None
        records.append(
            EpisodeRecord(
                episode_dir=episode_dir,
                session_name=episode_dir.parent.name,
                num_frames=len(df),
                task=task,
                task_index_hint=task_index_hint,
            )
        )
    return records


def discover_task_names(episodes_dir: Path | str, records: list[EpisodeRecord]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for session_dir in discover_session_dirs(episodes_dir):
        for task in _session_tasks(session_dir):
            if task not in seen:
                seen.add(task)
                ordered.append(task)
    for _hint, task in sorted(
        [(record.task_index_hint, record.task) for record in records if record.task_index_hint is not None],
        key=lambda item: int(item[0]),
    ):
        if task not in seen:
            seen.add(task)
            ordered.append(task)
    for task in sorted({record.task for record in records if record.task not in seen}):
        seen.add(task)
        ordered.append(task)
    return ordered or ["go_left", "go_right"]


def split_sessions(records: list[EpisodeRecord], split: str, val_ratio: float, seed: int) -> list[EpisodeRecord]:
    if split == "all":
        return list(records)
    sessions = sorted({record.session_name for record in records})
    if len(sessions) <= 1:
        return list(records) if split == "train" else []
    rng = random.Random(seed)
    rng.shuffle(sessions)
    val_count = max(1, int(round(len(sessions) * val_ratio)))
    val_sessions = set(sessions[-val_count:])
    if split == "train":
        return [record for record in records if record.session_name not in val_sessions]
    return [record for record in records if record.session_name in val_sessions]


def load_episode_frames(video_path: Path, image_size: tuple[int, int]) -> list[np.ndarray]:
    if av is None:
        raise RuntimeError("PyAV is required. Install with `/workspace/isaaclab/_isaac_sim/python.sh -m pip install av`.")
    width, height = image_size
    frames: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            frame = frame.reformat(width=width, height=height, format="rgb24")
            frames.append(np.asarray(frame.to_ndarray(), dtype=np.uint8))
    return frames


def load_episode_actions(parquet_path: Path) -> np.ndarray:
    df = pd.read_parquet(parquet_path, columns=["action"])
    actions = np.asarray(df["action"].tolist(), dtype=np.float32)
    if actions.shape[1] == DEFAULT_ACTION_DIM:
        return actions
    padded = np.zeros((actions.shape[0], DEFAULT_ACTION_DIM), dtype=np.float32)
    padded[:, : min(actions.shape[1], DEFAULT_ACTION_DIM)] = actions[:, :DEFAULT_ACTION_DIM]
    return padded


class EpisodeCache:
    def __init__(self, image_size: tuple[int, int], max_items: int = 8):
        self.image_size = image_size
        self.max_items = max_items
        self.frames: OrderedDict[Path, list[np.ndarray]] = OrderedDict()
        self.actions: OrderedDict[Path, np.ndarray] = OrderedDict()

    def get(self, record: EpisodeRecord) -> tuple[list[np.ndarray], np.ndarray]:
        key = record.episode_dir
        if key not in self.frames:
            self.frames[key] = load_episode_frames(key / "video.mp4", self.image_size)
            self.actions[key] = load_episode_actions(key / "data.parquet")
            if len(self.frames[key]) != len(self.actions[key]):
                raise ValueError(f"Frame/action count mismatch in {key}")
        self.frames.move_to_end(key)
        self.actions.move_to_end(key)
        if len(self.frames) > self.max_items:
            self.frames.popitem(last=False)
            self.actions.popitem(last=False)
        return self.frames[key], self.actions[key]


class ACTEpisodeDataset(Dataset):
    def __init__(
        self,
        records: list[EpisodeRecord],
        task_names: list[str],
        *,
        image_size: tuple[int, int] = (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT),
        chunk_size: int = DEFAULT_ACTION_CHUNK_SIZE,
        augment: bool = False,
        max_frames_per_episode: int = 0,
    ):
        self.records = list(records)
        self.task_names = list(task_names)
        self.task_to_index = {task: index for index, task in enumerate(self.task_names)}
        self.image_size = image_size
        self.chunk_size = chunk_size
        self.augment = augment
        self.max_frames_per_episode = max_frames_per_episode

        samples: list[SampleIndex] = []
        for episode_idx, record in enumerate(self.records):
            n = record.num_frames
            if max_frames_per_episode > 0 and n > max_frames_per_episode:
                indices = [int(round(i * (n - 1) / (max_frames_per_episode - 1))) for i in range(max_frames_per_episode)]
                indices = sorted(set(indices))
            else:
                indices = list(range(n))
            for frame_idx in indices:
                samples.append(SampleIndex(episode_idx=episode_idx, frame_idx=frame_idx))
        self.samples = samples
        self.cache = EpisodeCache(image_size=image_size, max_items=max(8, min(64, len(records) or 8)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        record = self.records[sample.episode_idx]
        frames, actions = self.cache.get(record)
        frame = Image.fromarray(frames[sample.frame_idx])
        if self.augment:
            frame = TF.adjust_brightness(frame, random.uniform(0.80, 1.20))
            frame = TF.adjust_contrast(frame, random.uniform(0.80, 1.20))
            frame = TF.adjust_saturation(frame, random.uniform(0.75, 1.25))
            frame = TF.adjust_hue(frame, random.uniform(-0.08, 0.08))
            w, h = frame.size
            crop_frac = random.uniform(0.88, 1.00)
            cw, ch = int(w * crop_frac), int(h * crop_frac)
            i = random.randint(0, h - ch)
            j = random.randint(0, w - cw)
            frame = TF.resized_crop(frame, i, j, ch, cw, (h, w))
        image = TF.to_tensor(frame)
        if self.augment:
            noise = torch.randn_like(image) * 0.015
            image = (image + noise).clamp(0.0, 1.0)

        chunk = np.zeros((self.chunk_size, DEFAULT_ACTION_DIM), dtype=np.float32)
        for offset in range(self.chunk_size):
            source_idx = min(sample.frame_idx + offset, len(actions) - 1)
            chunk[offset] = actions[source_idx]
        return {
            "image": image,
            "action_chunk": torch.as_tensor(chunk, dtype=torch.float32),
            "task_index": torch.tensor(self.task_to_index[record.task], dtype=torch.long),
            "task": record.task,
            "session_name": record.session_name,
        }


class CachedACTDataset(Dataset):
    """ACT dataset backed by predecoded tensors instead of MP4/parquet files."""

    def __init__(self, cache_path: Path | str, *, augment: bool = False):
        self.cache_path = Path(cache_path)
        payload = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        self.images = payload["images"].contiguous()
        self.action_chunks = payload["action_chunks"].contiguous()
        self.task_indices = payload["task_indices"].long().contiguous()
        self.task_names = list(payload["task_names"])
        self.augment = augment
        if self.images.ndim != 4:
            raise ValueError(f"Expected cached images as NHWC uint8, got shape {tuple(self.images.shape)}")
        if self.action_chunks.ndim != 3:
            raise ValueError(f"Expected cached action chunks as NLC, got shape {tuple(self.action_chunks.shape)}")
        if len(self.images) != len(self.action_chunks) or len(self.images) != len(self.task_indices):
            raise ValueError(f"Cached tensor length mismatch in {self.cache_path}")

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image = self.images[index].permute(2, 0, 1).float().div(255.0)
        if self.augment:
            image = TF.adjust_brightness(image, random.uniform(0.92, 1.08))
            image = TF.adjust_contrast(image, random.uniform(0.92, 1.08))
            image = TF.adjust_saturation(image, random.uniform(0.92, 1.08))
        task_index = self.task_indices[index]
        return {
            "image": image,
            "action_chunk": self.action_chunks[index].float(),
            "task_index": task_index,
            "task": self.task_names[int(task_index.item())],
            "session_name": self.cache_path.stem,
        }


def cached_dataset_paths(cache_dir: Path | str) -> tuple[Path, Path, Path]:
    root = Path(cache_dir)
    return root / "train.pt", root / "val.pt", root / "metadata.json"


def has_cached_datasets(cache_dir: Path | str) -> bool:
    train_path, val_path, metadata_path = cached_dataset_paths(cache_dir)
    return train_path.exists() and val_path.exists() and metadata_path.exists()


def build_cached_datasets(cache_dir: Path | str, *, augment: bool = True) -> tuple[CachedACTDataset, CachedACTDataset, list[str]]:
    train_path, val_path, metadata_path = cached_dataset_paths(cache_dir)
    metadata = _read_json(metadata_path)
    task_names = [str(task) for task in metadata.get("task_names", [])]
    train_ds = CachedACTDataset(train_path, augment=augment)
    val_ds = CachedACTDataset(val_path, augment=False)
    if not task_names:
        task_names = list(train_ds.task_names)
    return train_ds, val_ds, task_names


def build_datasets(
    episodes_dir: Path | str,
    *,
    image_size: tuple[int, int] = (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT),
    chunk_size: int = DEFAULT_ACTION_CHUNK_SIZE,
    val_ratio: float = 0.2,
    seed: int = 42,
    augment: bool = True,
    max_frames_per_episode: int = 0,
) -> tuple[ACTEpisodeDataset, ACTEpisodeDataset, list[str]]:
    records = discover_episodes(episodes_dir)
    task_names = discover_task_names(episodes_dir, records)
    train_records = split_sessions(records, "train", val_ratio, seed)
    val_records = split_sessions(records, "val", val_ratio, seed)
    return (
        ACTEpisodeDataset(
            train_records, task_names,
            image_size=image_size, chunk_size=chunk_size, augment=augment,
            max_frames_per_episode=max_frames_per_episode,
        ),
        ACTEpisodeDataset(
            val_records, task_names,
            image_size=image_size, chunk_size=chunk_size, augment=False,
            max_frames_per_episode=max_frames_per_episode,
        ),
        task_names,
    )
