import math
import os
import random
import numpy as np
import torch
import torch.nn.functional as F

VV_MEAN = -10.136466953246543
VV_STD = 5.013093346521575
VH_MEAN = -17.314963656896122
VH_STD = 6.202284981963544
EPS32 = np.finfo(np.float32).eps

def preprocess_linear_sigma(x):
    """Frozen G4-02 path. Channel axis must be -3 and channel order VV,VH."""
    a = np.asarray(x, dtype=np.float32).copy()
    if a.ndim < 3 or a.shape[-3] != 2:
        raise ValueError("Expected (...,2,H,W) or (2,H,W) in VV,VH order")
    invalid = (~np.isfinite(a)) | (a <= 0) | (a > 1e3)
    a[invalid] = 0.0
    np.maximum(a, EPS32, out=a)
    a = 10.0 * np.log10(a)
    means = np.array([VV_MEAN, VH_MEAN], dtype=np.float32)
    stds = np.array([VV_STD, VH_STD], dtype=np.float32)
    shape = [1] * a.ndim
    shape[-3] = 2
    a = (a - means.reshape(shape)) / stds.reshape(shape)
    return a.astype(np.float32, copy=False), invalid

def semantic_teacher_targets(logits):
    p = torch.softmax(logits, dim=1)
    pN, pP, pF = p[:,0:1], p[:,1:2], p[:,2:3]
    mP = torch.relu(pP - pF)
    qR = pF * (1.0 - mP)
    wW = 1.0 - pN
    entropy = -torch.xlogy(p, p).sum(dim=1, keepdim=True)
    reliability = 1.0 - entropy / math.log(3.0)
    qH = pF * (pF >= pP).to(pF.dtype)
    return {"p_N":pN,"p_P":pP,"p_F":pF,"q_R":qR,"w_W":wW,
            "reliability":reliability,"q_H":qH}

def binary_soft_dice(logits, target, valid, eps=1e-6):
    p = torch.sigmoid(logits)
    y = target.float()
    v = valid.float()
    if p.shape != y.shape or p.shape != v.shape:
        raise ValueError("binary_soft_dice shape mismatch")
    dims = tuple(range(1, p.ndim))
    inter = (p*y*v).sum(dims)
    denom = (p*v).sum(dims) + (y*v).sum(dims)
    return 1.0 - ((2.0*inter + eps)/(denom + eps)).mean()

def multiclass_soft_dice(logits, target, valid, eps=1e-6):
    if logits.ndim != 4 or logits.shape[1] != 3:
        raise ValueError("Expected Bx3xHxW logits")
    p = torch.softmax(logits, dim=1)
    y = F.one_hot(target.clamp(0,2), 3).permute(0,3,1,2).float()
    v = valid.float().unsqueeze(1)
    inter = (p*y*v).sum((2,3))
    denom = (p*v).sum((2,3)) + (y*v).sum((2,3))
    return 1.0 - ((2.0*inter + eps)/(denom + eps)).mean()

def seed_everything(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state

def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda_all" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])

def flood_probability(logits, condition):
    if condition == "A5" or condition == "TEACHER":
        return torch.softmax(logits, dim=1)[:,2:3]
    return torch.sigmoid(logits)
