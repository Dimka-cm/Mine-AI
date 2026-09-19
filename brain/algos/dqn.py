"""Double + Dueling DQN с прioritized-lite буфером."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model import CraftBrain, batch_obs
from ..spaces import N_ACTIONS


@dataclass
class DQNConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 50_000
    warmup: int = 500
    target_sync: int = 500
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000
    grad_clip: float = 5.0


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buf: Deque[tuple] = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done, next_mask=None) -> None:
        if next_mask is None:
            next_mask = np.ones(N_ACTIONS, dtype=bool)
        self.buf.append((obs, action, reward, next_obs, done, next_mask))

    def sample(self, n: int):
        batch = random.sample(self.buf, min(n, len(self.buf)))
        obs, act, rew, nobs, done, nmask = zip(*batch)
        return (list(obs), np.array(act), np.array(rew, dtype=np.float32),
                list(nobs), np.array(done, dtype=np.float32),
                np.stack(nmask))

    def __len__(self) -> int:
        return len(self.buf)


class DQNTrainer:
    def __init__(self, model: CraftBrain, cfg: DQNConfig | None = None,
                 device: str = "cpu") -> None:
        self.cfg = cfg or DQNConfig()
        self.device = device
        self.online = model.to(device)
        self.target = CraftBrain(model.cfg).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt = torch.optim.AdamW(self.online.parameters(), lr=self.cfg.lr)
        self.buffer = ReplayBuffer(self.cfg.buffer_size)
        self.steps = 0

    def epsilon(self) -> float:
        c = self.cfg
        frac = min(1.0, self.steps / c.eps_decay_steps)
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def act(self, obs: Dict[str, np.ndarray],
            mask: np.ndarray | None = None) -> int:
        """
        eps-greedy выбор действия.

        mask — булев вектор допустимых действий из env.action_mask().
        Он отсекает физически невозможные ходы (взять предмет, которого нет),
        но НЕ прячет содержательные ошибки: класть не в тот слот агент
        по-прежнему может и получает за это штраф.
        """
        self.steps += 1
        if mask is None:
            mask = np.ones(N_ACTIONS, dtype=bool)
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            valid = np.arange(N_ACTIONS)
        if random.random() < self.epsilon():
            return int(random.choice(valid))
        with torch.no_grad():
            q = self.online.q_values(batch_obs([obs], self.device)).squeeze(0)
        q[~torch.as_tensor(mask, device=self.device)] = -1e9
        return int(q.argmax().item())

    def learn(self) -> float | None:
        if len(self.buffer) < max(self.cfg.warmup, self.cfg.batch_size):
            return None
        obs, act, rew, nobs, done, nmask = self.buffer.sample(self.cfg.batch_size)
        b_obs = batch_obs(obs, self.device)
        b_nobs = batch_obs(nobs, self.device)
        act_t = torch.as_tensor(act, dtype=torch.long, device=self.device)
        rew_t = torch.as_tensor(rew, device=self.device)
        done_t = torch.as_tensor(done, device=self.device)

        q = self.online.q_values(b_obs).gather(1, act_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: действие выбираем online-сетью, оцениваем target-сетью.
            next_q_online = self.online.q_values(b_nobs)
            mask_t = torch.as_tensor(nmask, device=self.device)
            next_q_online = next_q_online.masked_fill(~mask_t, -1e9)
            next_a = next_q_online.argmax(dim=1, keepdim=True)
            next_q = self.target.q_values(b_nobs).gather(1, next_a).squeeze(1)
            tgt = rew_t + self.cfg.gamma * next_q * (1.0 - done_t)

        loss = F.smooth_l1_loss(q, tgt)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.opt.step()

        if self.steps % self.cfg.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())
