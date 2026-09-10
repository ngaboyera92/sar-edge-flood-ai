import math
import random
import numpy as np
import torch
import torch.nn.functional as F

VV_MEAN=-10.136466953246543
VV_STD=5.013093346521575
VH_MEAN=-17.314963656896122
VH_STD=6.202284981963544
EPS32=np.finfo(np.float32).eps


def preprocess_linear_sigma(x):
    """Exact G4-02 path for arrays shaped (...,2,H,W) or (2,H,W)."""
    a=np.asarray(x,dtype=np.float32).copy()
    invalid=(~np.isfinite(a)) | (a<=0) | (a>1e3)
    a[invalid]=0.0
    np.maximum(a, EPS32, out=a)
    a=10.0*np.log10(a)
    means=np.array([VV_MEAN,VH_MEAN],dtype=np.float32)
    stds=np.array([VV_STD,VH_STD],dtype=np.float32)
    axis=-3
    shape=[1]*a.ndim
    shape[axis]=2
    a=(a-means.reshape(shape))/stds.reshape(shape)
    return a.astype(np.float32), invalid


def semantic_teacher_targets(logits):
    p=torch.softmax(logits,dim=1)
    pN,pP,pF=p[:,0:1],p[:,1:2],p[:,2:3]
    mP=torch.relu(pP-pF)
    qR=pF*(1.0-mP)
    wW=1.0-pN
    pc=p.clamp_min(1e-12)
    entropy=-(pc*pc.log()).sum(dim=1,keepdim=True)
    reliability=1.0-entropy/math.log(3.0)
    qH=pF*(pF>=pP).to(pF.dtype)
    return {"p_N":pN,"p_P":pP,"p_F":pF,"q_R":qR,"w_W":wW,"reliability":reliability,"q_H":qH}


def binary_soft_dice(logits,target,valid,eps=1e-6):
    p=torch.sigmoid(logits)
    y=target.float()
    v=valid.float()
    dims=tuple(range(1,p.ndim))
    inter=(p*y*v).sum(dims)
    denom=(p*v).sum(dims)+(y*v).sum(dims)
    return 1.0-((2*inter+eps)/(denom+eps)).mean()


def multiclass_soft_dice(logits,target,valid,eps=1e-6):
    p=torch.softmax(logits,dim=1)
    y=F.one_hot(target.clamp(0,2),3).permute(0,3,1,2).float()
    v=valid.float().unsqueeze(1)
    inter=(p*y*v).sum((2,3))
    denom=(p*v).sum((2,3))+(y*v).sum((2,3))
    return 1.0-((2*inter+eps)/(denom+eps)).mean()


def kd_loss(student_logits,teacher_logits,mode,beta=0.3,eps=1e-6,valid=None):
    if mode=='none':
        return student_logits.new_tensor(0.0)
    s=torch.sigmoid(student_logits)
    t=semantic_teacher_targets(teacher_logits.detach())
    if valid is None:
        valid=torch.ones_like(s,dtype=torch.bool)
    v=valid.float()
    if mode=='ordinary_probability_mse':
        target=t['p_F']; w=torch.ones_like(target)
    elif mode=='reliability_weighted':
        target=t['p_F']; w=t['reliability']
    elif mode=='scope_only':
        target=t['p_F']; w=t['w_W']
    elif mode=='rectification_only':
        target=t['q_R']; w=torch.ones_like(target)
    elif mode=='semantic_rectification':
        target=t['q_R']; w=t['w_W']
    elif mode=='hard_stable_water_gate':
        target=t['q_H']; w=t['w_W']
    else:
        raise ValueError(mode)
    num=(w*v*(s-target).square()).sum()
    den=(w*v).sum()+eps
    return beta*num/den


def dataset_binary_iou_update(counts,logits,target,valid,threshold=0.5):
    pred=torch.sigmoid(logits)>threshold
    y=target.bool(); v=valid.bool()
    counts['tp'] += int((pred & y & v).sum())
    counts['fp'] += int((pred & ~y & v).sum())
    counts['fn'] += int((~pred & y & v).sum())
    return counts


def dataset_binary_iou(counts):
    den=counts['tp']+counts['fp']+counts['fn']
    return counts['tp']/den if den else 1.0


def seed_everything(seed):
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
