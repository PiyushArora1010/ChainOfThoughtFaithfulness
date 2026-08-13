import os
import numpy as np
import random
from numpy.random import RandomState
from numpy.random import seed as seednp

from module.models import Model

import torch
import torch.distributed as dist

def print0(*args, **kwargs):
    if kwargs.get("local_rank", 1) == 0:
        kwargs.pop("local_rank", None)
        if not dist.is_available() or not dist.is_initialized():
            print(*args, **kwargs)
        elif dist.get_rank() == 0:
            print(*args, **kwargs)

def set_seed(seed: int) -> RandomState:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set to false for reproducibility, True to boost performance
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    random_state = random.getstate()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return random_state

def get_language_model(model_tag, max_tokens=256, temperature=0.7, devices=None):
    return Model(name=model_tag, max_tokens=max_tokens, temperature=temperature, devices=devices)

class EMA:
    def __init__(self, value=None, alpha=0.99):
        self.alpha = alpha
        self.value = value

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * new_value

    def __call__(self):
        return self.value