"""NumPy-only client for the existing isolated eSEN worker.

Used by JAX baselines without installing Torch/fairchem in the JAX environment.
Each request/response pair is serialized. Raw-attempt accounting distinguishes
requests from acknowledged attempts; a timeout never implies zero computation.
"""
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import tempfile
import threading
import time

import numpy as np


class NumpyEnergyOracle:
    def __init__(self, interpreter, worker, checkpoint, *, numbers, charge,
                 spin_multiplicity, device='cpu', timeout_seconds=60., batch_size=1,
                 stderr_path=None, audit_repeats=False):
        self.condition={'numbers':list(map(int,numbers)),'charge':int(charge),
                        'spin_multiplicity':int(spin_multiplicity)}
        if not isinstance(batch_size,int) or batch_size<1 or not np.isfinite(timeout_seconds) or timeout_seconds<=0:
            raise ValueError('Require a positive worker batch and finite timeout')
        self.timeout=float(timeout_seconds);self.requested_evaluations=0;self.evaluated=0
        self.callback_requests=0;self._lock=threading.Lock();self._buffer=b''
        self._repeat_values={} if audit_repeats else None
        self.repeat_audit={'enabled':bool(audit_repeats),'compared_structures':0,
            'maximum_energy_difference_eV':0.,'maximum_force_difference_eV_A':0.}
        self.stderr=(open(stderr_path,'w+b') if stderr_path is not None else tempfile.TemporaryFile(dir='/tmp'))
        environment=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        environment.setdefault('XDG_CACHE_HOME','/tmp/bgfm_oracle_cache')
        environment.setdefault('MPLCONFIGDIR','/tmp/bgfm_oracle_mpl')
        self.process=subprocess.Popen([str(interpreter),'-u',str(worker),'--checkpoint',str(checkpoint),
            '--device',device,'--batch-size',str(batch_size)],stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=self.stderr,bufsize=0,env=environment)
        try:
            if self._receive().get('ready') is not True:raise RuntimeError('Oracle handshake failed')
        except Exception:
            self.close();raise

    def _receive(self):
        deadline=time.monotonic()+self.timeout
        while True:
            # Drain our own byte buffer before selecting the file descriptor;
            # TextIO readline can otherwise hide a complete buffered response.
            if b'\n' in self._buffer:
                line,self._buffer=self._buffer.split(b'\n',1)
                if line.startswith(b'BGFM_ORACLE_JSON '):
                    return json.loads(line[len(b'BGFM_ORACLE_JSON '):])
                continue
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('Energy oracle response timed out')
            ready,_,_=select.select([self.process.stdout],[],[],remaining)
            if not ready:raise TimeoutError('Energy oracle response timed out')
            chunk=os.read(self.process.stdout.fileno(),65536)
            if not chunk:
                self.stderr.seek(0)
                raise RuntimeError('Oracle exited: '+self.stderr.read()[-2000:].decode(errors='replace'))
            self._buffer+=chunk

    def evaluate(self, positions):
        x=np.asarray(positions,dtype=np.float64)
        if x.ndim!=3 or x.shape[1:]!=(len(self.condition['numbers']),3) or len(x)<1 or not np.isfinite(x).all():
            raise ValueError('Require finite [batch, declared atoms,3] positions')
        with self._lock:
            self.callback_requests+=1;self.requested_evaluations+=len(x)
            request=json.dumps({**self.condition,'positions':x.tolist()},allow_nan=False).encode()+b'\n'
            pending=memoryview(request)
            while pending:
                sent=self.process.stdin.write(pending)
                if not sent:raise RuntimeError('Oracle request pipe closed')
                pending=pending[sent:]
            response=self._receive()
            attempted=int(response.get('attempted_evaluations',0))
            if not 0<=attempted<=len(x):raise RuntimeError('Invalid acknowledged oracle count')
            self.evaluated+=attempted
            if response.get('ok') is not True:raise RuntimeError(response.get('error','Oracle request failed'))
            energy=np.asarray(response['energies_eV'],dtype=np.float64)
            force=np.asarray(response['forces_eV_A'],dtype=np.float64)
            if attempted!=len(x) or energy.shape!=(len(x),) or force.shape!=x.shape or not np.isfinite(energy).all() or not np.isfinite(force).all():
                raise RuntimeError('Invalid oracle energy, force or acknowledged count')
            if self._repeat_values is not None:
                for geometry,value,gradient in zip(x,energy,force):
                    digest=hashlib.sha256(np.ascontiguousarray(geometry).tobytes()).hexdigest()
                    if digest in self._repeat_values:
                        previous_energy,previous_force=self._repeat_values[digest]
                        self.repeat_audit['compared_structures']+=1
                        self.repeat_audit['maximum_energy_difference_eV']=max(
                            self.repeat_audit['maximum_energy_difference_eV'],float(abs(value-previous_energy)))
                        self.repeat_audit['maximum_force_difference_eV_A']=max(
                            self.repeat_audit['maximum_force_difference_eV_A'],float(np.max(np.abs(gradient-previous_force))))
                    else:self._repeat_values[digest]=(float(value),gradient.copy())
            return energy,force

    def evaluate_chunked(self, positions, *, max_request=32):
        x=np.asarray(positions,dtype=np.float64)
        if not isinstance(max_request,int) or max_request<1 or x.ndim!=3 or len(x)<1:
            raise ValueError('Require positive request bound and nonempty batch')
        values=[self.evaluate(x[start:start+max_request]) for start in range(0,len(x),max_request)]
        return np.concatenate([v[0] for v in values]),np.concatenate([v[1] for v in values])

    def close(self):
        if hasattr(self,'process'):
            if self.process.stdin is not None and not self.process.stdin.closed:self.process.stdin.close()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
            if self.process.stdout is not None:self.process.stdout.close()
        if hasattr(self,'stderr'):self.stderr.close()

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
