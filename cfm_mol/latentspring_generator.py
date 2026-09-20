"""Inference-only shared-EGNN LatentSpring bundle, without training/oracle imports."""
import hashlib,json,time
from pathlib import Path
import torch
from . import matched_egnn as base,connectivity_feedback as feedback
from .physical_connection import make_physical_connection
from .matched_physical_connection import PhysicalFieldTransform
from .hydrogen_completion import complete


class LatentSpringGenerator:
    def __init__(self,bundle,*,device='cpu',upstream=None):
        self.bundle=Path(bundle);manifest=json.loads((self.bundle/'manifest.json').read_text())
        if manifest['format']!='latentspring_egnn_inference_v1':raise ValueError('Unsupported bundle format')
        for name,h in manifest['files'].items():
            if hashlib.sha256((self.bundle/name).read_bytes()).hexdigest()!=h:raise ValueError('Bundle checksum mismatch: '+name)
        self.manifest=manifest;self.device=device;self.upstream=upstream;self.loaded={};self.heads={}

    def _head(self,fit,family):
        key=f'{family}_s{fit}'
        if key not in self.manifest['models']:raise ValueError('Head fit/family not present in this bundle')
        if key not in self.heads:
            h=torch.load(self.bundle/self.manifest['models'][key]['head'],map_location='cpu',weights_only=False)
            head=make_physical_connection(**h['configuration']).to(self.device);head.load_state_dict(h['state_dict'],strict=True);head.eval().requires_grad_(False)
            if base.state_hash(head)!=h['state_sha256']:raise ValueError('Loaded head state mismatch')
            self.heads[key]=head
        return self.heads[key]

    def _load(self,fit,family):
        key=f'{family}_s{fit}'
        if key not in self.manifest['models']:raise ValueError('Fit/family not present in this bundle')
        if key in self.loaded:return self.loaded[key]
        files=self.manifest['models'][key];p=torch.load(self.bundle/files['parent'],map_location='cpu',weights_only=False);h=torch.load(self.bundle/files['head'],map_location='cpu',weights_only=False);d=torch.load(self.bundle/files['hydrogen'],map_location='cpu',weights_only=False)
        spec=dict(p['spec']);hspec=dict(d['spec'])
        if self.upstream:spec['upstream']=hspec['upstream']=str(Path(self.upstream).resolve())
        model=base.initialize(spec,self.device)
        if spec.get('context')=='distance':feedback.install(model)
        model.load_state_dict(p['state_dict'],strict=True);model.eval().requires_grad_(False)
        head=self._head(fit,family)
        decoder=base.initialize(hspec,self.device);decoder.load_state_dict(d['state_dict'],strict=True);decoder.eval().requires_grad_(False)
        for obj,record in [(model,p),(head,h),(decoder,d)]:
            if base.state_hash(obj)!=record['state_sha256']:raise ValueError('Loaded model state mismatch')
        source=base.HarmonicSource(spec['edge_log_width']);context=feedback.GeometryContext(source,'distance') if spec.get('context')=='distance' else None
        self.loaded[key]=(model,head,decoder,spec,hspec,source,context);return self.loaded[key]

    @torch.no_grad()
    def generate(self,numbers,*,charge=0,spin_multiplicity=1,samples=16,seed=0,stream=0,fit=0,family='fm',head_family=None,hydrogen=True):
        if type(charge) is not int or type(spin_multiplicity) is not int or charge!=0 or spin_multiplicity!=1:raise ValueError('These checkpoints cover neutral singlets')
        if not isinstance(numbers,list) or not 2<=len(numbers)<=200 or any(type(z) is not int for z in numbers):raise ValueError('Require 2--200 integer atomic numbers')
        if type(samples) is not int or samples<1 or type(seed) is not int or seed<0:raise ValueError('Require a positive sample count and nonnegative integer seed')
        if type(stream) is not int or stream<0:raise ValueError('Require a nonnegative integer condition stream')
        if (sum(numbers)-charge-spin_multiplicity+1)%2:raise ValueError('Composition and electronic state have incompatible electron parity')
        model,head,decoder,spec,hspec,source,context=self._load(fit,family)
        head_family=family if head_family is None else head_family
        if head_family!=family:head=self._head(fit,head_family)
        if head.configuration['atomic_numbers']!=spec['atomic_numbers']:raise ValueError('Incompatible head vocabulary')
        if not {1,6}<=set(numbers) or not set(numbers)<=set(spec['atomic_numbers']):raise ValueError('Unsupported element composition')
        transform=PhysicalFieldTransform(model,spec,head,4.,strength_limit=4.);x0=[];parents=[];outputs=[];masks=[];hc=0;he=0;clock=time.perf_counter();core=[0]
        hook=model.dynamics.egnn.register_forward_hook(lambda *_:core.__setitem__(0,core[0]+1))
        try:
            for start in range(0,samples,8):
                count=min(8,samples-start);local_seed=seed*1000003+stream*100003+start
                if context:parent,initial=feedback.sample(model,numbers,spec,source,context,local_seed,count,128,field_transform=transform)
                else:parent,initial=base.sample(model,numbers,spec['kind'],spec,source,local_seed,count,128,field_transform=transform)
                z=torch.tensor(numbers,device=parent.device)[None].expand(count,-1)
                if hydrogen:
                    final,info=complete(decoder,parent,z,{'network_spec':hspec},mode='molecule',start_time=0.,steps=4,velocity_cap=2.)
                    hc+=info['network_calls'];he+=info['network_example_calls'];mask=info['changed']
                else:final=parent.clone();mask=torch.zeros(count,dtype=torch.bool,device=parent.device)
                x0.append(initial.cpu().double());parents.append(parent.cpu().double());outputs.append(final.cpu().double());masks.append(mask.cpu())
            if torch.device(self.device).type=='cuda':torch.cuda.synchronize()
        finally:hook.remove()
        batches=(samples+7)//8;assert core[0]==128*batches and transform.calls==batches*(64 if family=='fm' else 128)
        return dict(condition=dict(numbers=numbers,charge=charge,spin_multiplicity=spin_multiplicity),positions=torch.cat(outputs),
            parent_positions=torch.cat(parents),initial_positions=torch.cat(x0),hydrogen_changed=torch.cat(masks),
            costs=dict(attempted=samples,returned=samples,backbone_batch_calls=core[0],physical_head_batch_calls=transform.calls,
                hydrogen_batch_calls=hc,hydrogen_example_calls=he,oracle_queries=0,geometry_optimizer_steps=0,seconds=time.perf_counter()-clock),
            settings=dict(fit=fit,family=family,head_training_family=head_family,seed=seed,stream=stream,batch_size=8,physical_strength=4.,hydrogen_readout=hydrogen))
