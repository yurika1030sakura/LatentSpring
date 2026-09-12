import torch
from cfm_mol.source_force_screen import SourceForceScreen
from scripts.research.train_gate_distillation import per_edge_loss
from test_source_force_screen import force_pair


def test_actual_teacher_loss_has_sensitive_correct_parameter_gradients():
    torch.manual_seed(26011);row=force_pair();model=SourceForceScreen('neural').double()
    with torch.no_grad():
        model.coefficients.fill_(.001);model.encoder.head.weight.normal_(0,.01);model.encoder.head.bias.fill_(-.2)
    def objective():return per_edge_loss(model,row,'teacher',.0038653,128.,17.25,.0001)
    heads=[model.coefficients,model.encoder.head.weight];gradients=torch.autograd.grad(objective(),heads)
    for w,g in zip(heads,gradients):
        ix=int(g.abs().argmax());assert abs(float(g.flatten()[ix]))>1e-12;h=1e-6
        with torch.no_grad():w.flatten()[ix]+=h
        a=float(objective())
        with torch.no_grad():w.flatten()[ix]-=2*h
        b=float(objective())
        with torch.no_grad():w.flatten()[ix]+=h
        assert abs((a-b)/(2*h)-float(g.flatten()[ix]))<1e-6
    assert float(per_edge_loss(model,{'valid':False},'teacher',.003,128.,17.25,.0001))==0


def test_shared_index_sampling_preserves_each_declared_population_objective():
    values=torch.tensor([0.,.2,-.1,.8],dtype=torch.float64,requires_grad=True)
    selection=values.new_tensor([.1,.2,.3,.4])
    for weights in [values.new_full((4,),1/4),values.new_full((4,),1/2)]:
        estimated=(selection*(weights/selection)*values).sum()
        direct=(weights*values).sum()
        torch.testing.assert_close(estimated,direct,atol=1e-14,rtol=0)
        torch.testing.assert_close(torch.autograd.grad(estimated,values,retain_graph=True)[0],weights)
