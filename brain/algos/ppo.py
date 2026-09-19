"""PPO с GAE — вторая голова того же мозга."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model import CraftBrain, batch_obs


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 64
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    grad_clip: float = 0.5
    rollout: int = 512


class RolloutBuffer:
    def __init__(self) -> None:
        self.obs: List[dict] = []
        self.act: List[int] = []
        self.logp: List[float] = []
        self.rew: List[float] = []
        self.val: List[float] = []
        self.done: List[float] = []

    def add(self, obs, a, lp, r, v, d) -> None:
        self.obs.append(obs); self.act.append(a); self.logp.append(lp)
        self.rew.append(r); self.val.append(v); self.done.append(float(d))

    def clear(self) -> None:
        self.__init__()

    def __len__(self) -> int:
        return len(self.act)


class PPOTrainer:
    def __init__(self, model: CraftBrain, cfg: PPOConfig | None = None,
                 device: str = "cpu") -> None:
        self.cfg = cfg or PPOConfig()
        self.device = device
        self.model = model.to(device)
        self.opt = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr)
        self.buf = RolloutBuffer()

    def act(self, obs: Dict[str, np.ndarray], mask: np.ndarray | None = None):
        """Сэмплирование действия из политики с учётом маски допустимых ходов."""
        with torch.no_grad():
            logits, value = self.model.policy_value(batch_obs([obs], self.device))
            if mask is not None:
                m = torch.as_tensor(mask, device=self.device).unsqueeze(0)
                logits = logits.masked_fill(~m, -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(value.item())

    def _gae(self, last_value: float) -> tuple:
        c = self.cfg
        n = len(self.buf)
        adv = np.zeros(n, dtype=np.float32)
        last = 0.0
        vals = self.buf.val + [last_value]
        for t in reversed(range(n)):
            nonterm = 1.0 - self.buf.done[t]
            delta = self.buf.rew[t] + c.gamma * vals[t + 1] * nonterm - vals[t]
            last = delta + c.gamma * c.lam * nonterm * last
            adv[t] = last
        ret = adv + np.array(self.buf.val, dtype=np.float32)
        return adv, ret

    def update(self, last_value: float = 0.0) -> dict:
        c = self.cfg
        if len(self.buf) == 0:
            return {}
        adv, ret = self._gae(last_value)
        adv_t = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8), device=self.device)
        ret_t = torch.as_tensor(ret, device=self.device)
        act_t = torch.as_tensor(np.array(self.buf.act), dtype=torch.long, device=self.device)
        old_lp = torch.as_tensor(np.array(self.buf.logp, dtype=np.float32), device=self.device)
        obs_all = self.buf.obs
        n = len(self.buf)
        idx = np.arange(n)
        stats = {"pi_loss": 0.0, "vf_loss": 0.0, "entropy": 0.0, "n": 0}

        for _ in range(c.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, c.minibatch):
                mb = idx[start:start + c.minibatch]
                b = batch_obs([obs_all[i] for i in mb], self.device)
                logits, values = self.model.policy_value(b)
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(act_t[mb])
                ratio = torch.exp(lp - old_lp[mb])
                a = adv_t[mb]
                pi_loss = -torch.min(ratio * a,
                                     torch.clamp(ratio, 1 - c.clip, 1 + c.clip) * a).mean()
                vf_loss = F.mse_loss(values, ret_t[mb])
                ent = dist.entropy().mean()
                loss = pi_loss + c.vf_coef * vf_loss - c.ent_coef * ent

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
                self.opt.step()

                stats["pi_loss"] += float(pi_loss.item())
                stats["vf_loss"] += float(vf_loss.item())
                stats["entropy"] += float(ent.item())
                stats["n"] += 1
        self.buf.clear()
        k = max(1, stats["n"])
        return {k2: v / k for k2, v in stats.items() if k2 != "n"}
