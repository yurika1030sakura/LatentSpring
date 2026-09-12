"""Exact normalized vMF teacher-to-student KL for conditional angular guides."""
import torch
from cfm_mol.spherical_proposal import vmf_log_normalizer


def vmf_teacher_kl(teacher,student):
    if teacher.shape!=student.shape or teacher.ndim!=2 or teacher.shape[1]!=3:
        raise ValueError('Matched [batch,3] angular parameters required')
    if not torch.isfinite(teacher).all() or not torch.isfinite(student).all():
        raise ValueError('Finite teacher and student parameters required')
    teacher=teacher.detach();k=teacher.norm(dim=1);safe=k.clamp_min(1e-12)
    regular=1/torch.tanh(safe)-1/safe
    series=k/3-k.pow(3)/45+2*k.pow(5)/945
    length=torch.where(k<1e-3,series,regular)
    expected_direction=length[:,None]*teacher/safe[:,None]
    return vmf_log_normalizer(teacher)-vmf_log_normalizer(student)+((teacher-student)*expected_direction).sum(1)
