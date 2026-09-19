"""
ОГРАНИЧЕНИЕ ПО ЯДРАМ CPU.

Обучение по умолчанию сжирает все ядра и ноутбук перестаёт отзываться:
видео дёргается, браузер тормозит. Здесь жёстко резервируем ядра под
пользователя, чтобы машина оставалась живой.

Работает на двух уровнях, и оба нужны:
  1. привязка процесса к конкретным ядрам (affinity) — ОС физически не
     пустит его на зарезервированные;
  2. ограничение числа потоков в torch/numpy — иначе библиотеки создадут
     по потоку на каждое ядро системы и будут драться за время.

Одного torch.set_num_threads() недостаточно: OpenMP и MKL читают свои
переменные окружения, а выставлять их нужно ДО импорта numpy и torch.
"""

from __future__ import annotations

import os
from typing import List, Optional


def reserve_cores(n_workers: int = 5, verbose: bool = True) -> List[int]:
    """
    Оставляет процессу первые n_workers ядер, остальные — пользователю.

    Вызывать САМОЙ ПЕРВОЙ строкой в скрипте, до импорта numpy и torch:

        from brain.cpu_limit import reserve_cores
        reserve_cores(5)          # 5 ядер нам, остальные пользователю
        import torch              # уже увидит ограничение

    Возвращает список ядер, на которых разрешено работать.
    """
    total = os.cpu_count() or 1
    n = max(1, min(int(n_workers), total))
    cores = list(range(n))

    # 1) Переменные окружения для математических библиотек. Обязательно до
    #    первого импорта numpy/torch — потом они уже не читаются.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)

    # 2) Привязка к ядрам. Есть только на Linux; на Windows делается иначе.
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, set(cores))
        except OSError:
            pass
    else:
        _windows_affinity(cores)

    if verbose:
        free = total - n
        print(f"[cpu] всего ядер: {total} | обучению: {n} "
              f"| свободно вам: {free}")
    return cores


def _windows_affinity(cores: List[int]) -> None:
    """Привязка к ядрам на Windows — через psutil, если он установлен."""
    try:
        import psutil
        psutil.Process().cpu_affinity(cores)
    except Exception:
        # psutil нет — ограничения по потокам всё равно сработают.
        pass


def apply_torch_limits(n_workers: Optional[int] = None) -> None:
    """
    Ограничить потоки уже импортированного torch.

    Нужно, если reserve_cores() позвали поздно: переменные окружения
    опоздали, но это ещё поможет.
    """
    try:
        import torch
    except ImportError:
        return
    if n_workers is not None:
        n = int(n_workers)
    elif hasattr(os, "sched_getaffinity"):
        n = len(os.sched_getaffinity(0))
    else:
        n = os.cpu_count() or 1
    # Нельзя просить у torch больше потоков, чем разрешено ядер: он их
    # создаст, и они будут драться за одно и то же время.
    if hasattr(os, "sched_getaffinity"):
        n = min(n, len(os.sched_getaffinity(0)))
    torch.set_num_threads(int(n))
    # Межоперационный параллелизм тоже ограничиваем: иначе torch плодит
    # пулы поверх уже ограниченных потоков.
    try:
        torch.set_num_interop_threads(max(1, int(n) // 2))
    except RuntimeError:
        # Бросает, если пул уже создан — значит, ограничение поздно, ладно.
        pass


def child_initializer(cores: List[int]) -> None:
    """
    Инициализатор для процессов-воркеров (multiprocessing.Pool).

    Каждый воркер прибивается к ОДНОМУ ядру из списка — так они не мешают
    друг другу и не расползаются на зарезервированные пользователю ядра.
    """
    import multiprocessing as mp
    try:
        idx = mp.current_process()._identity[0] - 1
    except (AttributeError, IndexError):
        idx = 0
    core = cores[idx % len(cores)]
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {core})
        except OSError:
            pass
    else:
        _windows_affinity([core])
    try:
        import torch
        torch.set_num_threads(1)     # воркер = одно ядро = один поток
    except ImportError:
        pass
