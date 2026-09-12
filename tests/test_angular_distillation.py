import math
import torch
from cfm_mol.angular_distillation import vmf_teacher_kl
from cfm_mol.spherical_proposal import vmf_sample,vmf_log_prob


def test_vmf_kl_matches_sampled_density_ratio_and_uniform_limit():
    teacher=torch.tensor([[0.,0.,2.],[0.,0.,1024.]],dtype=torch.float64)
    student=torch.tensor([[.5,0.,1.],[10.,0.,1000.]],dtype=torch.float64)
    exact=vmf_teacher_kl(teacher,student)
    for i in range(2):
        t=teacher[i:i+1].expand(32768,-1);s=student[i:i+1].expand_as(t)
        draws,_=vmf_sample(t,generator=torch.Generator().manual_seed(24931+i))
        estimate=(vmf_log_prob(draws,t)-vmf_log_prob(draws,s)).mean()
        assert abs(float(estimate-exact[i]))<.02
    value=vmf_teacher_kl(torch.zeros(1,3,dtype=torch.float64),teacher[:1])
    torch.testing.assert_close(value,torch.tensor([math.log(math.sinh(2)/2)],dtype=torch.float64))


def test_identical_concentrated_teacher_has_zero_kl_and_finite_stationary_gradient():
    teacher=torch.tensor([[0.,0.,1024.],[0.,0.,0.]],dtype=torch.float64)
    student=teacher.clone().requires_grad_();loss=vmf_teacher_kl(teacher,student).sum();loss.backward()
    assert abs(float(loss))<1e-12
    assert torch.isfinite(student.grad).all() and float(student.grad.abs().max())<1e-10
