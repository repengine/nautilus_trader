"""
Masked time modeling pretraining (minimal scaffolding).

This module provides a lightweight pretraining routine for sequence encoders on
L0/L1 features using a masked reconstruction objective. The goal is to
initialize encoders prior to BCE fine‑tuning with L2/L3, improving convergence
and representation quality.

This is intentionally conservative and self‑contained. It trains a small GRU‑
based autoencoder to reconstruct randomly masked inputs. The resulting state
dict can be used to warm‑start compatible layers in downstream models.

Example:
    >>> import numpy as np
    >>> from ml.training.teacher.pretrain_mtm import PretrainConfig, MTMPretrainer
    >>> X = np.random.randn(1000, 30, 16).astype(np.float32)  # 1000 samples, 30 steps, 16 feats
    >>> cfg = PretrainConfig(input_dim=16, hidden_dim=32, seq_len=30, mask_prob=0.15, epochs=2)
    >>> pre = MTMPretrainer(cfg)
    >>> state_path = pre.fit_and_save(X, out_dir="/tmp/mtm_pretrain")
    >>> isinstance(state_path, str)
    True

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt


try:  # pragma: no cover - heavy dependency guard
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for masked time modeling pretraining") from exc


@dataclass(frozen=True)
class PretrainConfig:
    """
    Configuration for masked time modeling pretraining.

    Attributes:
        input_dim: Number of input features per timestep.
        hidden_dim: Hidden size for the GRU encoder/decoder.
        seq_len: Sequence length.
        mask_prob: Probability of masking each feature at each timestep.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Optimizer learning rate.
        seed: Optional RNG seed for reproducibility.

    """

    input_dim: int
    hidden_dim: int
    seq_len: int
    mask_prob: float = 0.15
    epochs: int = 5
    batch_size: int = 128
    learning_rate: float = 3e-4
    seed: int | None = 42


class _GRUAutoencoder(nn.Module):  # pragma: no cover - exercised in runtime path
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x, mask: (B, T, F)
        x_masked = x * (1.0 - mask)
        h, _ = self.encoder(x_masked)
        z, _ = self.decoder(h)
        y = self.proj(z)
        # Predict only masked positions
        loss = torch.nn.functional.mse_loss(y * mask, x * mask)
        return loss


class MTMPretrainer:
    """
    Masked time modeling pretrainer.

    This trains a small GRU autoencoder to reconstruct masked inputs.

    Args:
        cfg: Pretraining configuration.

    """

    def __init__(self, cfg: PretrainConfig) -> None:
        self.cfg = cfg
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)
        self.model = _GRUAutoencoder(cfg.input_dim, cfg.hidden_dim)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

    def fit_and_save(self, X: npt.NDArray[np.float32], out_dir: str | Path) -> str:
        """
        Train on ``X`` and save the state dict.

        Args:
            X: Training data of shape (N, T, F) with dtype float32.
            out_dir: Output directory to write ``pretrained_state.pt``.

        Returns:
            Path to the saved state dict.

        """
        assert X.dtype == np.float32 and X.ndim == 3
        N, T, F = X.shape
        if T != self.cfg.seq_len or F != self.cfg.input_dim:
            raise ValueError("X shape does not match configured seq_len/input_dim")
        ds = torch.utils.data.TensorDataset(torch.from_numpy(X))
        dl = torch.utils.data.DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True)
        self.model.train()
        for _ in range(self.cfg.epochs):
            for (xb,) in dl:
                xb = xb.float()
                # Mask: Bernoulli mask over (B,T,F)
                mask = (torch.rand_like(xb) < self.cfg.mask_prob).float()
                loss = self.model(xb, mask)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        out = Path(out_dir) / "pretrained_state.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(out))
        return str(out)
