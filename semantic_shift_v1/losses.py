import torch.nn.functional as F
from .core import binary_soft_dice,multiclass_soft_dice,kd_loss

MODE_MAP={'A0':'ordinary_probability_mse','B3':'reliability_weighted','A1':'scope_only','A2':'rectification_only','A3':'semantic_rectification','A4':'hard_stable_water_gate','B0':'none','A5':'none'}


def teacher_loss(logits,label,valid):
    flat=valid.view(-1)
    ce=F.cross_entropy(logits.permute(0,2,3,1).reshape(-1,3)[flat],label.view(-1)[flat])
    return ce+multiclass_soft_dice(logits,label,valid)


def student_loss(condition,student_logits,label,valid,teacher_logits=None,beta=0.3):
    if condition=='A5':
        return teacher_loss(student_logits,label,valid)
    y=(label==2).float().unsqueeze(1)
    v=valid.unsqueeze(1)
    bce=F.binary_cross_entropy_with_logits(student_logits[v],y[v])
    sup=bce+binary_soft_dice(student_logits,y,v)
    if condition=='B0':
        return sup
    if teacher_logits is None:
        raise ValueError('teacher_logits required')
    return sup+kd_loss(student_logits,teacher_logits,MODE_MAP[condition],beta=beta,valid=v)
