"""Reproducibility helpers — set seed deterministic cho mọi run (CLAUDE.md mục 8).

Mọi script training PHẢI gọi ``set_seed(seed)`` trước khi tạo model/data, và
truyền ``seed_worker`` + ``make_generator(seed)`` vào DataLoader để worker cũng
deterministic.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Cố định seed cho random / numpy / torch (CPU + CUDA).

    ``deterministic=True`` bật cudnn deterministic và tắt benchmark — chậm hơn
    chút nhưng tái lập được (yêu cầu của dự án: log trung thực, reproduce được).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Khởi tạo seed cho mỗi DataLoader worker (truyền vào ``worker_init_fn``)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Generator cho DataLoader (truyền vào ``generator=``) để shuffle ổn định."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
