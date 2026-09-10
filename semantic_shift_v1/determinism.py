"""Deterministic execution controls frozen by G4-04."""
import random
import numpy as np
import torch
from torch.utils.data import get_worker_info

from .core import seed_everything, capture_rng_state, restore_rng_state


def make_train_generator(seed):
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def seed_worker(worker_id):
    """Seed worker Python/NumPy and any dataset-local augmentation RNG."""
    del worker_id
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, "set_worker_seed"):
        info.dataset.set_worker_seed(worker_seed)


def capture_recovery_rng(train_generator=None):
    state = {"global": capture_rng_state()}
    if train_generator is not None:
        state["train_generator"] = train_generator.get_state()
    return state


def restore_recovery_rng(state, train_generator=None):
    restore_rng_state(state["global"])
    if train_generator is not None and "train_generator" in state:
        train_generator.set_state(state["train_generator"])
