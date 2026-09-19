#!/usr/bin/env python3
"""
ОБУЧЕНИЕ ЗРЕНИЯ: научить глаза агента читать положение в пространстве.

Это НЕ обучение с подкреплением и НЕ путь до алмазного меча. Здесь решается
одна отдельная задача: по картинке понять, где ты стоишь и куда смотришь.

Почему отдельно. В RL сигнал приходит редко и с задержкой — награда за меч
ничего не говорит о том, правильно ли агент видит стену слева. А тут ответ
известен точно: среда сама знает координаты, с которых нарисован кадр. Это
обучение с учителем, и оно сходится в десятки раз быстрее.

Чему учим (всё читается ТОЛЬКО по пикселям, никаких служебных чисел на входе):
    x, z          — где агент стоит
    наклон головы — смотрит вверх, прямо или вниз
    сторона света — куда повёрнут (4 класса)
    4 расстояния  — до стены впереди, справа, сзади, слева
    видимость     — есть ли в кадре дерево / камень / железо / алмаз

Потом обученные свёртки переносятся в PixelBrain, и RL стартует уже зрячим,
а не с нуля.

Запуск:
    python3 pretrain_vision.py --samples 6000 --epochs 12
    python3 pretrain_vision.py --quick          # быстрая проверка
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from brain.cpu_limit import apply_torch_limits, reserve_cores
from brain.env.mc_env import WORLD_D, WORLD_W, MinecraftCraftEnv
from brain.env.render import Raycaster
from brain.rewards.db import RewardDB
from brain.rewards.engine import RewardEngine
from brain.rewards.rules import seed as seed_rules
from brain.spaces import BLOCK_ID

# Куда смотрит агент: SOUTH=+Z, WEST=-X, NORTH=-Z, EAST=+X
FORWARD = {0: (0, 1), 1: (-1, 0), 2: (0, -1), 3: (1, 0)}

# Что агент должен уметь замечать в кадре
VISIBLE_BLOCKS = ["oak_log", "stone", "iron_ore", "diamond_ore"]

TARGETS = ["x", "z", "наклон", "стена впереди", "стена справа",
           "стена сзади", "стена слева"] + [f"видит {b}" for b in VISIBLE_BLOCKS]


# --------------------------------------------------------------------------
# СБОР ДАННЫХ
# --------------------------------------------------------------------------
def wall_distances(ax: int, az: int, facing: int) -> list:
    """
    Расстояния до четырёх стен в системе взгляда агента: вперёд, вправо,
    назад, влево. Именно относительные, а не абсолютные: "стена слева"
    полезна агенту, "стена на западе" — нет, её ещё надо переводить.
    """
    out = []
    for k in range(4):
        dx, dz = FORWARD[(facing + k) % 4]
        if dx > 0:
            v = (WORLD_W - 1 - ax) / (WORLD_W - 1)
        elif dx < 0:
            v = ax / (WORLD_W - 1)
        elif dz > 0:
            v = (WORLD_D - 1 - az) / (WORLD_D - 1)
        else:
            v = az / (WORLD_D - 1)
        out.append(float(v))
    return out


def visible_flags(world: np.ndarray, ax: int, az: int, facing: int) -> list:
    """
    Есть ли блок в поле зрения — грубо, по сектору 90 градусов впереди.
    Учим не «есть ли он в мире», а «видно ли его отсюда»: иначе сеть будет
    угадывать по памяти мира, а не смотреть на картинку.
    """
    fdx, fdz = FORWARD[facing]
    flags = []
    for name in VISIBLE_BLOCKS:
        bid = BLOCK_ID[name]
        seen = 0.0
        xs, zs = np.where(world[:, 2, :] == bid)
        for bx, bz in zip(xs, zs):
            dx, dz = int(bx) - ax, int(bz) - az
            dist = (dx * dx + dz * dz) ** 0.5
            if dist < 0.5 or dist > 10:
                continue
            # скалярное произведение с направлением взгляда
            if (dx * fdx + dz * fdz) / dist > 0.55:
                seen = 1.0
                break
        flags.append(seen)
    return flags


def collect(n: int, rc: Raycaster, env: MinecraftCraftEnv, rng, h: int, w: int,
            new_world_every: int = 25, verbose: bool = True):
    """
    Набирает выборку: кадр -> правильные ответы.

    Мир пересоздаётся каждые new_world_every кадров. Это принципиально:
    на одном мире сеть выучит конкретный ландшафт наизусть, а нам нужно
    умение ЧИТАТЬ сцену, которое перенесётся на новый мир.
    """
    X = np.empty((n, 3, h, w), dtype=np.uint8)
    Y = np.empty((n, len(TARGETS)), dtype=np.float32)
    FA = np.empty(n, dtype=np.int64)
    t0 = time.perf_counter()
    for i in range(n):
        if i % new_world_every == 0:
            env.reset()
        world = env.world
        # свободная клетка внутри ограды
        for _ in range(60):
            ax = int(rng.integers(1, WORLD_W - 1))
            az = int(rng.integers(1, WORLD_D - 1))
            if world[ax, 2, az] == 0:
                break
        facing = int(rng.integers(0, 4))
        pitch = int(rng.integers(-1, 2))
        X[i] = rc.render(world, ax, 2, az, facing, pitch, env.mobs).transpose(2, 0, 1)
        Y[i] = ([ax / (WORLD_W - 1), az / (WORLD_D - 1), (pitch + 1) / 2.0]
                + wall_distances(ax, az, facing)
                + visible_flags(world, ax, az, facing))
        FA[i] = facing
        if verbose and (i + 1) % 500 == 0:
            el = time.perf_counter() - t0
            print(f"  кадров {i + 1}/{n}  ({el:.0f} с, осталось ~{el / (i + 1) * (n - i - 1):.0f} с)",
                  flush=True)
    return X, Y, FA


# --------------------------------------------------------------------------
# МОДЕЛЬ
# --------------------------------------------------------------------------
class VisionNet(nn.Module):
    """
    Свёрточная часть ровно такая же, как в PixelBrain, — иначе веса потом
    не перенести. Сверху временные головы: их после обучения выбрасываем,
    нужны только натренированные свёртки.
    """

    def __init__(self, h: int, w: int, c_in: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, 32, 8, stride=4), nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.SiLU(),
            # Позиция признака на экране обязана дожить до этого места,
            # поэтому здесь свёртка с шагом, а не усредняющий пулинг.
            nn.Conv2d(64, 64, 3, stride=2), nn.SiLU(),
            nn.Conv2d(64, 32, 1), nn.SiLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, c_in, h, w)).shape[1]
        self.n_flat = n_flat
        self.trunk = nn.Sequential(nn.Linear(n_flat, 512), nn.SiLU())
        self.reg = nn.Linear(512, len(TARGETS))   # координаты/стены/видимость
        self.fac = nn.Linear(512, 4)              # сторона света

    def forward(self, x):
        z = self.trunk(self.conv(x))
        return self.reg(z), self.fac(z)


# --------------------------------------------------------------------------
# ОБУЧЕНИЕ
# --------------------------------------------------------------------------
def run(args):
    reserve_cores(args.cores)
    apply_torch_limits()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    h, w = args.height, args.width
    tmp = tempfile.mkdtemp()
    db = RewardDB(str(Path(tmp) / "pretrain.db"))
    seed_rules(db)
    env = MinecraftCraftEnv(RewardEngine(db), max_steps=90, seed=args.seed,
                            mobs=args.mobs)
    rc = Raycaster(w, h, fov=75.0, max_dist=12.0)
    rng = np.random.default_rng(args.seed)

    n_test = max(200, args.samples // 6)
    # Кадр 426x240x3 = 307 КБ. Держать в ОЗУ десятки тысяч нельзя:
    # 6000 кадров это уже 1.8 ГБ. Предупреждаем честно, до сбора.
    mb = (args.samples + n_test) * 3 * h * w / 1e6
    print(f"Сбор данных: {args.samples} обучающих + {n_test} проверочных, {w}x{h}")
    print(f"Потребуется ОЗУ под кадры: ~{mb:.0f} МБ")
    if mb > args.ram_limit:
        raise SystemExit(
            f"Слишком много: {mb:.0f} МБ > лимита {args.ram_limit} МБ.\n"
            f"Возьмите --samples поменьше (примерно "
            f"{int(args.ram_limit * 1e6 / (3 * h * w) / 1.17)}) "
            f"или поднимите --ram-limit, если памяти хватает.")
    Xtr, Ytr, Ftr = collect(args.samples, rc, env, rng, h, w)
    Xte, Yte, Fte = collect(n_test, rc, env, rng, h, w, verbose=False)
    print(f"В памяти: {(Xtr.nbytes + Xte.nbytes) / 1e6:.0f} МБ\n")

    model = VisionNet(h, w)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"Модель зрения: {n_par:,} параметров, flatten={model.n_flat:,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    Ytr_t, Ftr_t = torch.from_numpy(Ytr), torch.from_numpy(Ftr)
    n = len(Xtr)

    print(f"\nОбучение: {args.epochs} эпох по {n} кадров, батч {args.batch}")
    t_start = time.perf_counter()
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(n)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xb = torch.from_numpy(Xtr[idx]).float().div_(255.0)
            reg, fac = model(xb)
            loss = (F.mse_loss(reg, Ytr_t[idx])
                    + 0.5 * F.cross_entropy(fac, Ftr_t[idx]))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * len(idx)
        sched.step()
        el = time.perf_counter() - t_start
        print(f"  эпоха {ep + 1:2d}/{args.epochs}  потеря {tot / n:.4f}  "
              f"({el:.0f} с, осталось ~{el / (ep + 1) * (args.epochs - ep - 1):.0f} с)",
              flush=True)

    # ---- проверка ----
    model.eval()
    P, FP = [], []
    with torch.no_grad():
        for i in range(0, len(Xte), args.batch):
            xb = torch.from_numpy(Xte[i:i + args.batch]).float().div_(255.0)
            r, f = model(xb)
            P.append(r)
            FP.append(f.argmax(1))
    P = torch.cat(P).numpy()
    FP = torch.cat(FP).numpy()
    base = Ytr.mean(0)

    print("\n" + "=" * 66)
    print("ЧТО АГЕНТ НАУЧИЛСЯ ВИДЕТЬ (проверка на новых мирах)")
    print("=" * 66)
    print(f"{'признак':22s} {'ошибка':>9s} {'без обучения':>13s}  {'':>8s}")
    print("-" * 66)
    report = {}
    for j, name in enumerate(TARGETS):
        mae = float(np.abs(P[:, j] - Yte[:, j]).mean())
        bl = float(np.abs(base[j] - Yte[:, j]).mean())
        gain = 1 - mae / bl if bl > 1e-9 else 0.0
        mark = "хорошо" if gain > 0.45 else ("есть" if gain > 0.15 else "слабо")
        extra = ""
        if name in ("x", "z"):
            extra = f"{mae * (WORLD_W - 1):.2f} бл"
        print(f"{name:22s} {mae:9.4f} {bl:13.4f}  {mark:>8s} {extra}")
        report[name] = {"mae": round(mae, 4), "baseline": round(bl, 4)}
    acc = float((FP == Fte).mean())
    print(f"{'сторона света':22s} {acc * 100:8.0f}% {25:12d}%  "
          f"{'хорошо' if acc > 0.7 else 'слабо':>8s}")
    report["facing_acc"] = round(acc, 4)
    print("=" * 66)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"conv": model.conv.state_dict(), "width": w, "height": h,
                "targets": TARGETS, "report": report}, args.out)
    json.dump(report, open(Path(args.out).with_suffix(".json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"\nГлаза сохранены: {args.out}")
    print(f"Всего заняло: {(time.perf_counter() - t_start) / 60:.1f} мин")
    return report


def load_into_brain(brain, path: str = "vision.pt", verbose: bool = True) -> bool:
    """
    Переносит обученные глаза в боевую модель PixelBrain.

    Тонкость: pretrain учится на ОДНОМ кадре (3 канала), а агент видит стек
    из нескольких (12 каналов). Совпадает всё, кроме первого слоя. Его веса
    размножаем по стеку и делим на число кадров — так средняя яркость отклика
    сохраняется, и модель не получает на старте зашкаливающие активации.

    Возвращает True, если веса легли.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["conv"]
    target = brain.conv.state_dict()
    n_frames = brain.cfg.n_frames

    out = {}
    for k, v in sd.items():
        tv = target.get(k)
        if tv is None:
            continue
        if v.shape == tv.shape:
            out[k] = v
        elif k == "0.weight" and tv.shape[1] == v.shape[1] * n_frames:
            out[k] = v.repeat(1, n_frames, 1, 1) / n_frames
        elif verbose:
            print(f"  пропущен {k}: {tuple(v.shape)} -> {tuple(tv.shape)}")
    missing = [k for k in target if k not in out]
    brain.conv.load_state_dict({**target, **out})
    if verbose:
        rep = ck.get("report", {})
        print(f"Глаза загружены из {path}: перенесено {len(out)}/{len(target)} тензоров")
        if missing:
            print(f"  осталось случайными: {missing}")
        if "facing_acc" in rep:
            print(f"  сторона света на обучении: {rep['facing_acc'] * 100:.0f}%")
    return len(out) > 0


def main():
    ap = argparse.ArgumentParser(description="Обучение зрения агента (координаты по картинке)")
    ap.add_argument("--samples", type=int, default=6000, help="сколько кадров собрать")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--width", type=int, default=426)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--cores", type=int, default=5, help="лимит ядер CPU")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mobs", action="store_true", help="добавить мобов в кадр")
    ap.add_argument("--out", default="vision.pt")
    ap.add_argument("--ram-limit", type=int, default=1200,
                    help="сколько МБ ОЗУ можно занять кадрами")
    ap.add_argument("--quick", action="store_true", help="быстрая проверка на малой выборке")
    args = ap.parse_args()
    if args.quick:
        args.samples, args.epochs = 600, 4
    run(args)


if __name__ == "__main__":
    main()
