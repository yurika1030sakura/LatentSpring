"""Torch-side client for a pinned, isolated JAX EACF transport worker."""
import json
import os
from pathlib import Path
import select
import subprocess
import time
import torch


class EACFTransport:
    def __init__(self, interpreter, worker, run, upstream, *, log_path, timeout=600.):
        self.timeout = timeout
        self.buffer = b''
        self.requests = 0
        self.requested_transformations = 0
        self.completed_transformations = 0
        self.validation_transformations = 0
        self.seconds = 0.
        self.stderr = open(log_path, 'w+b')
        env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
            JAX_PLATFORMS='cpu', JAX_ENABLE_X64='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
            XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2')
        env['PYTHONPATH'] = str(Path(worker).resolve().parents[2])+':'+str(upstream)
        self.process = subprocess.Popen([str(interpreter), '-u', str(worker), '--run', str(run)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, bufsize=0, env=env)
        try:
            self.handshake = self.receive()
            if not self.handshake.get('ready') or self.handshake['backend'] != 'cpu':
                raise ValueError('Invalid isolated transport handshake')
            self.aux_scale = self.handshake['auxiliary_scale']
        except Exception:
            self.close()
            raise

    def receive(self):
        deadline = time.monotonic()+self.timeout
        while True:
            if b'\n' in self.buffer:
                line, self.buffer = self.buffer.split(b'\n', 1)
                if line.startswith(b'BGFM_EACF_JSON '):
                    return json.loads(line[len(b'BGFM_EACF_JSON '):])
                continue
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Transport response timed out; no restart or zero-cost inference')
            ready, _, _ = select.select([self.process.stdout], [], [], min(10., remaining))
            if not ready:
                continue
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                self.stderr.seek(0)
                raise RuntimeError('Transport exited: '+self.stderr.read()[-3000:].decode(errors='replace'))
            self.buffer += chunk

    def transform(self, x, auxiliary, directions, *, order=None, check_inverse=False, check_jacobian=False):
        request = dict(positions=x.detach().cpu().tolist(), auxiliary=auxiliary.detach().cpu().tolist(),
            directions=directions.cpu().tolist(), check_inverse=check_inverse, check_jacobian=check_jacobian)
        if order is not None:
            request['order'] = list(map(int, order))
        self.requests += 1
        self.requested_transformations += len(x)
        pending = memoryview((json.dumps(request, allow_nan=False)+'\n').encode())
        while pending:
            n = self.process.stdin.write(pending)
            if not n:
                raise RuntimeError('Transport request pipe closed')
            pending = pending[n:]
        response = self.receive()
        if not response.get('ok'):
            raise RuntimeError(response.get('error', 'Transport failed'))
        self.completed_transformations += response['transformations']
        self.validation_transformations += response.get('validation_transformations', 0)
        self.seconds += response['seconds']
        return {**response, **{key: torch.tensor(response[key], dtype=torch.float64)
                             for key in ['positions', 'auxiliary', 'log_volume']}}

    def close(self):
        if hasattr(self, 'process'):
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            if self.process.stdout:
                self.process.stdout.close()
        self.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
