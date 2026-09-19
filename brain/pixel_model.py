"""
МОДЕЛЬ С ПИКСЕЛЬНЫМ ЗРЕНИЕМ.

Две ветки, потом слияние:

  картинка  -> CNN (свёртки)      -> вектор 512
  предметы  -> эмбеддинги + MLP   -> вектор 256
                                      \\
                                       -> ствол -> головы DQN / PPO

CNN — архитектура из DQN Nature (Mnih et al., 2015), проверенная на Atari,
с поправкой на наш размер кадра. Свёртки хороши тем, что не зависят от
положения объекта: зомби слева и зомби справа опознаются одними и теми же
весами. Для полносвязной сети это были бы две разные задачи.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spaces import DENSE_DIM, GRID_SIZE, N_ACTIONS, N_ITEMS


@dataclass
class PixelConfig:
    """Настройки пиксельной модели."""
    width: int = 426
    height: int = 240
    n_frames: int = 4          # сколько кадров видит одновременно
    item_emb: int = 24
    cnn_out: int = 512
    sym_out: int = 256
    hidden: int = 512
    head: str = "both"         # 'dqn' | 'ppo' | 'both'
    n_actions: int = N_ACTIONS


class PixelBrain(nn.Module):
    """Мозг агента: глаза (CNN) + чтение предметов (эмбеддинги)."""

    def __init__(self, cfg: PixelConfig = PixelConfig()):
        super().__init__()
        self.cfg = cfg
        c_in = 3 * cfg.n_frames          # RGB на каждый кадр стека

        # --- ГЛАЗА --------------------------------------------------------
        # Шаги 4-2-1: первый слой грубо сжимает кадр, последние уточняют.
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, 32, 8, stride=4), nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.SiLU(),
            # ВАЖНО: здесь нельзя ставить AdaptiveAvgPool2d.
            # Усредняющий пулинг намеренно теряет ПОЗИЦИЮ признака на экране,
            # а агенту она нужна: где стена, слева или справа, — это и есть
            # ориентация в пространстве. Замер показал разницу наглядно:
            # с avgpool сеть определяла сторону света на 27% (случайно 25%),
            # со свёрткой шага 2 — на 84%.
            # Поэтому сжимаем ещё одной свёрткой (шаг 2 сохраняет карту
            # признаков 12x24), а число каналов режем 1x1-свёрткой:
            # 100 096 -> 9 216 чисел на входе полносвязной части.
            nn.Conv2d(64, 64, 3, stride=2), nn.SiLU(),
            nn.Conv2d(64, 32, 1), nn.SiLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.conv(
                torch.zeros(1, c_in, cfg.height, cfg.width)).shape[1]
        self.cnn_proj = nn.Sequential(
            nn.Linear(n_flat, cfg.cnn_out), nn.SiLU())

        # --- ЧТЕНИЕ ПРЕДМЕТОВ ---------------------------------------------
        self.item_emb = nn.Embedding(N_ITEMS, cfg.item_emb)
        # Позиция в сетке крафта важна: доска НАД палкой — это меч,
        # доска ПОД палкой — не рецепт. Эмбеддинг слота это кодирует.
        self.slot_pos = nn.Parameter(torch.zeros(1, GRID_SIZE, cfg.item_emb))
        self.grid_proj = nn.Linear(GRID_SIZE * cfg.item_emb, 128)
        self.dense_proj = nn.Linear(DENSE_DIM, 128)
        self.sym_trunk = nn.Sequential(
            nn.Linear(128 + 128 + cfg.item_emb, cfg.sym_out), nn.SiLU())

        # --- СЛИЯНИЕ ------------------------------------------------------
        self.trunk = nn.Sequential(
            nn.Linear(cfg.cnn_out + cfg.sym_out, cfg.hidden),
            nn.LayerNorm(cfg.hidden), nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.LayerNorm(cfg.hidden), nn.SiLU(),
        )

        # --- ГОЛОВЫ -------------------------------------------------------
        if cfg.head in ("dqn", "both"):
            # Dueling: отдельно "насколько хороша позиция" и "насколько
            # хорошо конкретное действие". Учится устойчивее обычного DQN.
            self.q_val = nn.Linear(cfg.hidden, 1)
            self.q_adv = nn.Linear(cfg.hidden, cfg.n_actions)
        if cfg.head in ("ppo", "both"):
            self.pi = nn.Linear(cfg.hidden, cfg.n_actions)
            self.v = nn.Linear(cfg.hidden, 1)

        self.apply(self._init)

    @staticmethod
    def _init(m):
        """Ортогональная инициализация — стандарт для RL, стабильнее."""
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.05)

    # -- прямой проход -----------------------------------------------------
    def encode(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.cnn_proj(self.conv(obs["pixels"]))

        g = self.item_emb(obs["grid"]) + self.slot_pos
        g = F.silu(self.grid_proj(g.flatten(1)))
        d = F.silu(self.dense_proj(obs["dense"]))
        h = self.item_emb(obs["held"].squeeze(-1))
        s = self.sym_trunk(torch.cat([g, d, h], dim=-1))

        return self.trunk(torch.cat([x, s], dim=-1))

    def q_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.encode(obs)
        v, a = self.q_val(z), self.q_adv(z)
        return v + a - a.mean(dim=-1, keepdim=True)

    def policy(self, obs: Dict[str, torch.Tensor]):
        z = self.encode(obs)
        return self.pi(z), self.v(z).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --- подготовка данных ----------------------------------------------------
def obs_to_tensor(obs: Dict[str, np.ndarray], device="cpu") -> Dict:
    """Одно наблюдение -> батч размера 1."""
    return {
        "pixels": torch.as_tensor(obs["pixels"], dtype=torch.float32,
                                  device=device).unsqueeze(0),
        "grid": torch.as_tensor(obs["grid"], dtype=torch.long,
                                device=device).unsqueeze(0),
        "dense": torch.as_tensor(obs["dense"], dtype=torch.float32,
                                 device=device).unsqueeze(0),
        "held": torch.as_tensor(obs["held"], dtype=torch.long,
                                device=device).unsqueeze(0),
    }


def batch_obs(items: List[Dict[str, np.ndarray]], device="cpu") -> Dict:
    """Список наблюдений -> один батч."""
    return {
        "pixels": torch.as_tensor(np.stack([o["pixels"] for o in items]),
                                  dtype=torch.float32, device=device),
        "grid": torch.as_tensor(np.stack([o["grid"] for o in items]),
                                dtype=torch.long, device=device),
        "dense": torch.as_tensor(np.stack([o["dense"] for o in items]),
                                 dtype=torch.float32, device=device),
        "held": torch.as_tensor(np.stack([o["held"] for o in items]),
                                dtype=torch.long, device=device),
    }
