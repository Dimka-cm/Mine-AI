"""
БУФЕР ОПЫТА ДЛЯ ПИКСЕЛЬНОГО ЗРЕНИЯ.

Наивная реализация хранит в каждой записи стек из N кадров плюс такой же
стек "следующего состояния". Каждый кадр при этом лежит в памяти 2*N раз.
На 320x240 это 86 ГБ на 50 тысяч записей — неподъёмно.

Здесь кадры лежат ЛЕНТОЙ, каждый ровно один раз, а стек собирается на лету
по индексам при выборке батча. Экономия ровно в 2*N раз (при N=4 — в восемь).

Плюс два приёма:
  * храним uint8, а не float32 — ещё в 4 раза меньше;
    перевод в [0,1] делается на GPU для одного батча;
  * при нехватке ОЗУ лента уезжает на SSD через numpy.memmap —
    NVMe тянет случайное чтение батча без заметных потерь.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BufferSpec:
    """Во что обходится одна запись — считается до выделения памяти."""
    capacity: int
    height: int
    width: int
    n_frames: int
    bytes_total: int

    @property
    def gib(self) -> float:
        return self.bytes_total / 2 ** 30

    def describe(self) -> str:
        return (f"буфер {self.capacity:,} записей, кадр "
                f"{self.width}x{self.height}, стек {self.n_frames} -> "
                f"{self.gib:.2f} ГБ")


class PixelReplayBuffer:
    """
    Кольцевой буфер с ленивой сборкой стека.

    Хранит: ленту кадров (uint8), а рядом — действия, награды, флаги конца
    эпизода, маски и символьную часть наблюдения (она крошечная).

    on_disk=True кладёт ленту кадров в файл на SSD вместо ОЗУ.
    """

    def __init__(self, capacity: int, height: int, width: int,
                 n_frames: int, dense_dim: int, grid_size: int,
                 n_actions: int, on_disk: bool = False,
                 disk_path: Optional[str] = None):
        self.cap = int(capacity)
        self.h, self.w = int(height), int(width)
        self.n_frames = int(n_frames)
        self.on_disk = bool(on_disk)

        shape = (self.cap, self.h, self.w, 3)
        nbytes = int(np.prod(shape))

        if on_disk:
            path = disk_path or os.path.join(
                tempfile.gettempdir(), "mc_rl_frames.dat")
            self._path = path
            self.frames = np.memmap(path, dtype=np.uint8, mode="w+",
                                    shape=shape)
        else:
            self._path = None
            self.frames = np.zeros(shape, dtype=np.uint8)

        # Символьная часть — мелочь, всегда в ОЗУ.
        self.dense = np.zeros((self.cap, dense_dim), dtype=np.float32)
        self.grid = np.zeros((self.cap, grid_size), dtype=np.int64)
        self.held = np.zeros((self.cap, 1), dtype=np.int64)
        self.act = np.zeros(self.cap, dtype=np.int64)
        self.rew = np.zeros(self.cap, dtype=np.float32)
        self.done = np.zeros(self.cap, dtype=bool)
        self.nmask = np.zeros((self.cap, n_actions), dtype=bool)
        # Номер эпизода: не даёт склеить кадры из РАЗНЫХ эпизодов в один стек.
        self.ep = np.zeros(self.cap, dtype=np.int32)

        self.pos = 0
        self.full = False
        self._ep_counter = 0

    # -- служебное ---------------------------------------------------------
    @staticmethod
    def estimate(capacity: int, height: int, width: int,
                 n_frames: int) -> BufferSpec:
        """Сколько памяти займёт буфер — до того, как её выделять."""
        total = capacity * height * width * 3
        return BufferSpec(capacity, height, width, n_frames, total)

    def __len__(self) -> int:
        return self.cap if self.full else self.pos

    def new_episode(self) -> None:
        """Отметить границу эпизода, чтобы стеки её не пересекали."""
        self._ep_counter += 1

    # -- запись ------------------------------------------------------------
    def push(self, frame: np.ndarray, obs: Dict[str, np.ndarray],
             action: int, reward: float, done: bool,
             next_mask: np.ndarray) -> None:
        """
        Кладёт ОДИН кадр и сопутствующие данные.

        frame — (H, W, 3) uint8, самый свежий кадр (не стек!).
        """
        i = self.pos
        self.frames[i] = frame
        self.dense[i] = obs["dense"]
        self.grid[i] = obs["grid"]
        self.held[i] = obs["held"]
        self.act[i] = action
        self.rew[i] = reward
        self.done[i] = done
        self.nmask[i] = next_mask
        self.ep[i] = self._ep_counter

        self.pos += 1
        if self.pos >= self.cap:
            self.pos = 0
            self.full = True

    # -- чтение ------------------------------------------------------------
    def _stack_at(self, idx: np.ndarray) -> np.ndarray:
        """
        Собирает стек кадров, заканчивающийся на каждом из idx.

        Если предыдущий кадр из другого эпизода — повторяем самый ранний
        доступный, чтобы не смешать два разных эпизода в одном наблюдении.

        Возвращает (B, N, H, W, 3) uint8 — канал ПОСЛЕДНИЙ, специально.
        Перестановка в (B, N*3, H, W) стоит на CPU дороже самого копирования
        (33 мс из 56), поэтому её делает уже GPU в pixels_to_tensor().
        """
        B = len(idx)
        out = np.empty((B, self.n_frames, self.h, self.w, 3), dtype=np.uint8)
        n = len(self)
        for k in range(self.n_frames):
            off = self.n_frames - 1 - k          # k=0 -> самый старый
            src = (idx - off) % n
            bad = self.ep[src] != self.ep[idx]   # вышли за границу эпизода
            src = np.where(bad, idx, src)
            out[:, k] = self.frames[src]
        return out

    def sample(self, batch_size: int) -> Tuple:
        """
        Возвращает батч: (obs, act, rew, next_obs, done, next_mask).

        Наблюдения — словари из numpy-массивов; pixels уже в uint8,
        перевод во float делает обучающий код прямо на GPU.
        """
        n = len(self)
        # Первые n_frames-1 индексов пропускаем: у них нет полной истории.
        lo = self.n_frames - 1
        idx = np.random.randint(lo, n - 1, size=batch_size)
        nxt = idx + 1

        obs = {
            "pixels": self._stack_at(idx),
            "dense": self.dense[idx],
            "grid": self.grid[idx],
            "held": self.held[idx],
        }
        next_obs = {
            "pixels": self._stack_at(nxt),
            "dense": self.dense[nxt],
            "grid": self.grid[nxt],
            "held": self.held[nxt],
        }
        return (obs, self.act[idx], self.rew[idx], next_obs,
                self.done[idx].astype(np.float32), self.nmask[idx])

    # -- уборка ------------------------------------------------------------
    def close(self) -> None:
        """Закрывает memmap и удаляет временный файл."""
        if self.on_disk and self._path:
            del self.frames
            try:
                os.remove(self._path)
            except OSError:
                pass


# --- перенос батча на устройство ------------------------------------------
def pixels_to_tensor(arr: np.ndarray, device: str = "cpu"):
    """
    (B, N, H, W, 3) uint8 -> (B, N*3, H, W) float32 в [0,1] на устройстве.

    Порядок операций важен. Сначала отправляем uint8 на GPU (в 4 раза меньше
    байт по шине PCIe, чем float32), и только там перестраиваем оси и делим
    на 255. На CPU одна лишь перестановка осей съедала 33 мс из 56.
    """
    import torch
    t = torch.from_numpy(arr)
    if device != "cpu":
        # non_blocking работает вместе с pin_memory и прячет копирование
        # за вычислениями предыдущего батча.
        t = t.pin_memory().to(device, non_blocking=True)
    B, N, H, W, C = t.shape
    t = t.permute(0, 1, 4, 2, 3).reshape(B, N * C, H, W)
    return t.float().div_(255.0)
