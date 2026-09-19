"""
"Пустая" (необученная) нейросеть агента.

Архитектура — общий энкодер + две головы, чтобы можно было переключаться
между DQN и PPO без перестройки весов:

    grid(9 слотов) --embed--> \
    voxels(27 блоков) --embed--> concat -> MLP -> features(256)
    dense(инвентарь/ориентация/цель) --/
                                          |-> Dueling Q-head   (DQN)
                                          |-> Policy + Value   (PPO)

Слои инициализируются ортогонально и НЕ загружают никаких предобученных
весов — модель действительно пустая и учится только на наградах.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spaces import (DENSE_DIM, GRID_SIZE, N_ACTIONS, N_BLOCKS, N_ENT_MAP,
                     N_ENTITIES, N_ITEMS, N_MAP, N_VOXELS)


def _ortho(layer: nn.Module, gain: float = np.sqrt(2)) -> nn.Module:
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.constant_(layer.bias, 0.0)
    return layer


@dataclass
class ModelConfig:
    item_emb: int = 24
    block_emb: int = 16
    hidden: int = 256
    layers: int = 2
    head: str = "both"    # 'dqn' | 'ppo' | 'both'
    dueling: bool = True


class CraftBrain(nn.Module):
    """Общий мозг агента."""

    def __init__(self, cfg: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg

        # Эмбеддинги: предмет в слоте и блок в вокселе.
        self.item_emb = nn.Embedding(N_ITEMS, c.item_emb)
        self.block_emb = nn.Embedding(N_BLOCKS, c.block_emb)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.block_emb.weight, std=0.02)

        # Сетка крафта: позиция важна -> добавляем обучаемый позиционный код.
        self.slot_pos = nn.Parameter(torch.zeros(GRID_SIZE, c.item_emb))
        nn.init.normal_(self.slot_pos, std=0.02)
        self.grid_proj = _ortho(nn.Linear(GRID_SIZE * c.item_emb, c.hidden // 2))

        # Воксели вокруг агента — ближнее 3D-зрение (5x5x5).
        self.vox_proj = _ortho(nn.Linear(N_VOXELS * c.block_emb, c.hidden // 2))

        # Дальнее 2D-зрение: карта блоков сверху (13x13) — куда идти.
        self.map_proj = _ortho(nn.Linear(N_MAP * c.block_emb, c.hidden // 2))

        # Карта сущностей: где мобы. Отдельный эмбеддинг — моб и блок
        # принципиально разные вещи, мешать их в один словарь нельзя.
        self.ent_emb = nn.Embedding(N_ENTITIES, c.block_emb)
        nn.init.normal_(self.ent_emb.weight, std=0.02)
        self.ent_proj = _ortho(nn.Linear(N_ENT_MAP * c.block_emb, c.hidden // 2))

        # Плотные признаки.
        self.dense_proj = _ortho(nn.Linear(DENSE_DIM, c.hidden // 2))
        self.held_emb = nn.Embedding(N_ITEMS, c.item_emb)
        nn.init.normal_(self.held_emb.weight, std=0.02)

        # grid + voxels + blockmap + entmap + dense + held
        trunk_in = (c.hidden // 2) * 5 + c.item_emb
        trunk = []
        d = trunk_in
        for _ in range(c.layers):
            trunk += [_ortho(nn.Linear(d, c.hidden)), nn.LayerNorm(c.hidden), nn.SiLU()]
            d = c.hidden
        self.trunk = nn.Sequential(*trunk)

        # --- голова DQN (dueling) ---
        if c.head in ("dqn", "both"):
            self.q_val = nn.Sequential(
                _ortho(nn.Linear(c.hidden, c.hidden // 2)), nn.SiLU(),
                _ortho(nn.Linear(c.hidden // 2, 1), gain=1.0))
            self.q_adv = nn.Sequential(
                _ortho(nn.Linear(c.hidden, c.hidden // 2)), nn.SiLU(),
                _ortho(nn.Linear(c.hidden // 2, N_ACTIONS), gain=1.0))

        # --- голова PPO ---
        if c.head in ("ppo", "both"):
            self.pi = nn.Sequential(
                _ortho(nn.Linear(c.hidden, c.hidden // 2)), nn.SiLU(),
                _ortho(nn.Linear(c.hidden // 2, N_ACTIONS), gain=0.01))
            self.vf = nn.Sequential(
                _ortho(nn.Linear(c.hidden, c.hidden // 2)), nn.SiLU(),
                _ortho(nn.Linear(c.hidden // 2, 1), gain=1.0))

    # ---------------- кодирование ----------------
    def encode(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        g = self.item_emb(obs["grid"]) + self.slot_pos       # (B,9,E)
        g = self.grid_proj(g.flatten(1))
        v = self.block_emb(obs["voxels"]).flatten(1)          # (B,125*E)
        v = self.vox_proj(v)
        bm = self.map_proj(self.block_emb(obs["blockmap"]).flatten(1))
        em = self.ent_proj(self.ent_emb(obs["entmap"]).flatten(1))
        d = self.dense_proj(obs["dense"])
        h = self.held_emb(obs["held"].squeeze(-1))
        x = torch.cat([F.silu(g), F.silu(v), F.silu(bm), F.silu(em),
                       F.silu(d), h], dim=-1)
        return self.trunk(x)

    # ---------------- головы ----------------
    def q_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        f = self.encode(obs)
        if not self.cfg.dueling:
            return self.q_adv(f)
        val = self.q_val(f)
        adv = self.q_adv(f)
        return val + adv - adv.mean(dim=-1, keepdim=True)

    def policy_value(self, obs: Dict[str, torch.Tensor]
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(obs)
        return self.pi(f), self.vf(f).squeeze(-1)

    # ---------------- удобные обёртки ----------------
    @torch.no_grad()
    def act_greedy(self, obs: Dict[str, torch.Tensor]) -> int:
        return int(self.q_values(obs).argmax(dim=-1).item())

    @torch.no_grad()
    def act_sample(self, obs: Dict[str, torch.Tensor],
                   temperature: float = 1.0) -> Tuple[int, float, float]:
        logits, value = self.policy_value(obs)
        dist = torch.distributions.Categorical(logits=logits / temperature)
        a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(value.item())

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def obs_to_tensor(obs: Dict[str, np.ndarray], device: str = "cpu"
                  ) -> Dict[str, torch.Tensor]:
    """numpy-наблюдение -> батч из одного элемента."""
    return {
        "grid": torch.as_tensor(obs["grid"], dtype=torch.long, device=device).unsqueeze(0),
        "voxels": torch.as_tensor(obs["voxels"], dtype=torch.long, device=device).unsqueeze(0),
        "blockmap": torch.as_tensor(obs["blockmap"], dtype=torch.long, device=device).unsqueeze(0),
        "entmap": torch.as_tensor(obs["entmap"], dtype=torch.long, device=device).unsqueeze(0),
        "dense": torch.as_tensor(obs["dense"], dtype=torch.float32, device=device).unsqueeze(0),
        "held": torch.as_tensor(obs["held"], dtype=torch.long, device=device).unsqueeze(0),
    }


def batch_obs(obs_list, device: str = "cpu") -> Dict[str, torch.Tensor]:
    return {
        "grid": torch.as_tensor(np.stack([o["grid"] for o in obs_list]),
                                dtype=torch.long, device=device),
        "voxels": torch.as_tensor(np.stack([o["voxels"] for o in obs_list]),
                                  dtype=torch.long, device=device),
        "blockmap": torch.as_tensor(np.stack([o["blockmap"] for o in obs_list]),
                                    dtype=torch.long, device=device),
        "entmap": torch.as_tensor(np.stack([o["entmap"] for o in obs_list]),
                                  dtype=torch.long, device=device),
        "dense": torch.as_tensor(np.stack([o["dense"] for o in obs_list]),
                                 dtype=torch.float32, device=device),
        "held": torch.as_tensor(np.stack([o["held"] for o in obs_list]),
                                dtype=torch.long, device=device),
    }
