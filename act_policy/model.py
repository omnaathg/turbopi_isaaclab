"""Language-conditioned ACT + CVAE model for TurboPi image policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch import nn

from . import (
    DEFAULT_ACTION_CHUNK_SIZE,
    DEFAULT_ACTION_DIM,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    EXPECTED_ACT_CVAE_PARAM_COUNT,
)


@dataclass(frozen=True)
class ACTCVAEConfig:
    image_width: int = DEFAULT_IMAGE_WIDTH
    image_height: int = DEFAULT_IMAGE_HEIGHT
    task_vocab_size: int = 2
    d_model: int = 64
    action_dim: int = DEFAULT_ACTION_DIM
    chunk_size: int = DEFAULT_ACTION_CHUNK_SIZE
    z_dim: int = 16
    cvae_hidden_dim: int = 64
    encoder_heads: int = 2
    decoder_heads: int = 2
    transformer_ffn_dim: int = 128
    dropout: float = 0.0
    expected_param_count: int = EXPECTED_ACT_CVAE_PARAM_COUNT


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LanguageConditionedACTCVAE(nn.Module):
    """ACT policy matching the requested 64 spatial + 1 language + 1 z token layout."""

    def __init__(self, config: ACTCVAEConfig | None = None):
        super().__init__()
        self.config = config or ACTCVAEConfig()
        cfg = self.config
        if cfg.image_width % 16 != 0 or cfg.image_height % 16 != 0:
            raise ValueError(
                f"image_width and image_height must be multiples of 16 (4 stride-2 conv blocks). "
                f"Got {cfg.image_width}x{cfg.image_height}."
            )
        num_spatial_tokens = (cfg.image_height // 16) * (cfg.image_width // 16)

        self.vision = nn.Sequential(
            ConvBlock(3, 16, kernel_size=5, stride=2, padding=2),
            ConvBlock(16, 32, kernel_size=3, stride=2, padding=1),
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=2, padding=1),
        )
        self.spatial_pos = nn.Parameter(torch.zeros(num_spatial_tokens, cfg.d_model))
        self.language_embedding = nn.Embedding(cfg.task_vocab_size, cfg.d_model)

        self.cvae_encoder = nn.Sequential(
            nn.Linear(cfg.chunk_size * cfg.action_dim + cfg.d_model, cfg.cvae_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.z_mu = nn.Linear(cfg.cvae_hidden_dim, cfg.z_dim)
        self.z_logvar = nn.Linear(cfg.cvae_hidden_dim, cfg.z_dim)
        self.z_projection = nn.Linear(cfg.z_dim, cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.encoder_heads,
            dim_feedforward=cfg.transformer_ffn_dim,
            dropout=cfg.dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.decoder_heads,
            dim_feedforward=cfg.transformer_ffn_dim,
            dropout=cfg.dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.action_queries = nn.Parameter(torch.zeros(cfg.chunk_size, cfg.d_model))
        self.action_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.d_model, cfg.action_dim),
            nn.Tanh(),
        )

        if cfg.expected_param_count > 0:
            base_count = self.parameter_count()
            calibration_count = cfg.expected_param_count - base_count
            if calibration_count < 0:
                raise ValueError(f"Base model has {base_count} params, above requested {cfg.expected_param_count}.")
            self.parameter_count_calibration = nn.Parameter(torch.zeros(calibration_count))
            final_count = self.parameter_count()
            if final_count != cfg.expected_param_count:
                raise ValueError(f"ACT parameter count mismatch: expected {cfg.expected_param_count}, got {final_count}.")

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def encode_actions(self, action_chunk: torch.Tensor, language_token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_actions = action_chunk.flatten(1)
        hidden = self.cvae_encoder(torch.cat([flat_actions, language_token], dim=1))
        mu = self.z_mu(hidden)
        logvar = self.z_logvar(hidden).clamp(min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return z, mu, logvar

    def forward(
        self,
        images: torch.Tensor,
        task_ids: torch.Tensor,
        action_chunk: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        expected = (3, self.config.image_height, self.config.image_width)
        if images.ndim != 4 or images.shape[1:] != expected:
            raise ValueError(f"Expected images [B,{expected[0]},{expected[1]},{expected[2]}], got {tuple(images.shape)}")
        if task_ids.ndim == 0:
            task_ids = task_ids.unsqueeze(0)
        task_ids = task_ids.long()
        batch_size = images.shape[0]
        language_token = self.language_embedding(task_ids)

        features = self.vision(images)
        spatial_tokens = features.flatten(2).transpose(1, 2) + self.spatial_pos.unsqueeze(0)
        if action_chunk is None:
            z = torch.zeros(batch_size, self.config.z_dim, device=images.device, dtype=images.dtype)
            mu = torch.zeros_like(z)
            logvar = torch.zeros_like(z)
        else:
            z, mu, logvar = self.encode_actions(action_chunk, language_token)
        z_token = self.z_projection(z).unsqueeze(1)

        memory_tokens = torch.cat([spatial_tokens, language_token.unsqueeze(1), z_token], dim=1)
        memory = self.transformer_encoder(memory_tokens)
        queries = self.action_queries.unsqueeze(0).expand(batch_size, -1, -1) + language_token.unsqueeze(1)
        decoded = self.transformer_decoder(queries, memory)
        action_pred = self.action_head(decoded)
        return {"action": action_pred, "mu": mu, "logvar": logvar}


def build_model(config: ACTCVAEConfig | None = None) -> LanguageConditionedACTCVAE:
    return LanguageConditionedACTCVAE(config=config)


def save_checkpoint(
    path: Path,
    model: LanguageConditionedACTCVAE,
    *,
    epoch: int,
    metrics: dict[str, float],
    extra: dict[str, object] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "metrics": metrics,
            "model_config": asdict(model.config),
            "model_state_dict": model.state_dict(),
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(
    path: Path,
    map_location: str | torch.device | None = None,
) -> tuple[LanguageConditionedACTCVAE, dict[str, object]]:
    payload = torch.load(Path(path), map_location=map_location)
    raw_config = payload.get("model_config", {})
    known = {field.name for field in fields(ACTCVAEConfig)}
    config = ACTCVAEConfig(**{key: value for key, value in raw_config.items() if key in known})
    model = LanguageConditionedACTCVAE(config)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload
