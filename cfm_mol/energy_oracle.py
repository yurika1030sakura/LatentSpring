"""Persistent subprocess oracle without mixing fairchem and DGL environments."""
import json
import os
from pathlib import Path
import select
import subprocess
import tempfile
import time

import torch


class EnergyOracle:
    def __init__(self, interpreter, worker, checkpoint, *, numbers, charge,
                 spin_multiplicity, device='cpu', timeout_seconds=60.):
        self.timeout=timeout_seconds;self.evaluated=0;self.requested_evaluations=0
        self.condition={'numbers':list(map(int,numbers)),'charge':int(charge),
                        'spin_multiplicity':int(spin_multiplicity)}
        self.stderr=tempfile.TemporaryFile(mode='w+t',dir='/tmp')
        environment=os.environ.copy()
        environment.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        environment.setdefault('XDG_CACHE_HOME','/tmp/bgfm_oracle_cache')
        environment.setdefault('MPLCONFIGDIR','/tmp/bgfm_oracle_mpl')
        self.process=subprocess.Popen([str(interpreter),'-u',str(worker),'--checkpoint',str(checkpoint),
            '--device',device],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,
            text=True,bufsize=1,env=environment)
        try:
            if self._receive().get('ready') is not True:raise RuntimeError('Oracle handshake failed')
        except Exception:
            self.close();raise

    def _receive(self):
        deadline=time.monotonic()+self.timeout
        while time.monotonic()<deadline:
            readable,_,_=select.select([self.process.stdout],[],[],max(0,deadline-time.monotonic()))
            if not readable:raise TimeoutError('Energy oracle response timed out')
            line=self.process.stdout.readline()
            if not line:
                self.stderr.seek(0)
                raise RuntimeError('Oracle exited: '+self.stderr.read()[-2000:])
            if line.startswith('BGFM_ORACLE_JSON '):return json.loads(line[len('BGFM_ORACLE_JSON '):])
        raise TimeoutError('Energy oracle response timed out')

    def evaluate(self, positions):
        positions=torch.as_tensor(positions).detach().cpu().double()
        if positions.ndim!=3 or not torch.isfinite(positions).all():raise ValueError('Invalid oracle positions')
        request={**self.condition,'positions':positions.tolist()}
        self.requested_evaluations+=len(positions)
        self.process.stdin.write(json.dumps(request,allow_nan=False)+'\n');self.process.stdin.flush()
        response=self._receive()
        self.evaluated+=int(response.get('attempted_evaluations',0))
        if response.get('ok') is not True:raise RuntimeError(response.get('error','Invalid oracle response'))
        energy=torch.tensor(response['energies_eV'],dtype=torch.float64)
        force=torch.tensor(response['forces_eV_A'],dtype=torch.float64)
        if energy.shape!=(len(positions),) or force.shape!=positions.shape or not torch.isfinite(energy).all() or not torch.isfinite(force).all():
            raise RuntimeError('Invalid oracle energy/force shape or value')
        return energy,force

    def close(self):
        if hasattr(self,'process'):
            if self.process.stdin is not None:self.process.stdin.close()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
            if self.process.stdout is not None:self.process.stdout.close()
        if hasattr(self,'stderr'):self.stderr.close()

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
