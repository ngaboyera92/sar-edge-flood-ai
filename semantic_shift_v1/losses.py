import torch
import torch.nn.functional as F
from .core import binary_soft_dice, multiclass_soft_dice, semantic_teacher_targets

def _masked_mean(x, valid):
    v = valid.bool()
    n = int(v.sum().item())
    if n == 0:
        raise RuntimeError("No valid pixels in batch")
    return x[v].mean()

def _normalized_weighted_mse(s, target, weight, valid, eps=1e-6):
    v = valid.float()
    num = (weight * v * (s-target).square()).sum()
    den = (weight * v).sum() + eps
    return num / den

def kd_term(condition, student_logits, teacher_logits, valid, eps=1e-6):
    if condition in {"B0","A5"}:
        return student_logits.new_tensor(0.0)
    if teacher_logits is None:
        raise ValueError("teacher_logits required for KD condition")
    s = torch.sigmoid(student_logits)
    t = semantic_teacher_targets(teacher_logits.detach())
    v = valid.bool()
    if condition == "A0":
        return _masked_mean((s-t["p_F"]).square(), v)
    if condition == "B3":
        return _normalized_weighted_mse(s,t["p_F"],t["reliability"],v,eps)
    if condition == "A1":
        return _normalized_weighted_mse(s,t["p_F"],t["w_W"],v,eps)
    if condition == "A2":
        return _masked_mean((s-t["q_R"]).square(), v)
    if condition == "A3":
        return _normalized_weighted_mse(s,t["q_R"],t["w_W"],v,eps)
    if condition == "A4":
        return _normalized_weighted_mse(s,t["q_H"],t["w_W"],v,eps)
    raise ValueError(f"Unknown condition {condition}")

def teacher_loss(logits, label, valid):
    flat = valid.reshape(-1).bool()
    if int(flat.sum().item()) == 0:
        raise RuntimeError("No valid pixels in batch")
    ce = F.cross_entropy(
        logits.permute(0,2,3,1).reshape(-1,3)[flat],
        label.reshape(-1)[flat],
        reduction="mean",
    )
    return ce + multiclass_soft_dice(logits,label,valid,eps=1e-6)

def student_loss(condition, student_logits, label, valid, teacher_logits=None, beta=0.3):
    if condition == "A5":
        return teacher_loss(student_logits,label,valid)
    y = (label == 2).float().unsqueeze(1)
    v = valid.bool().unsqueeze(1)
    if int(v.sum().item()) == 0:
        raise RuntimeError("No valid pixels in batch")
    bce = F.binary_cross_entropy_with_logits(student_logits[v], y[v], reduction="mean")
    sup = bce + binary_soft_dice(student_logits,y,v,eps=1e-6)
    if condition == "B0":
        return sup
    return sup + float(beta) * kd_term(condition,student_logits,teacher_logits,v,eps=1e-6)
