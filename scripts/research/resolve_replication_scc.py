"""Apply one uniform fixed-geometry SCC completion policy to every failed score."""
import argparse,hashlib,json,os,shutil,subprocess,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from cfm_mol.xtb_singlepoint import parse_singlepoint

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--prepare',action='store_true');a=p.parse_args();root=a.project.resolve();a.out=a.out.resolve();proto=root/'research/evidence/replication_scc_completion_v1.json'
    if a.prepare:
        assert not proto.exists();sources={};failures={}
        for campaign in ['seed_replication_v1','cross_generator_head_v1']:
            for si in range(5):
                folder=root/'runs'/campaign/(f'evaluation/s{si}' if campaign=='seed_replication_v1' else f's{si}')
                files=[folder/('parent_xtb/results.json' if campaign=='seed_replication_v1' else 'xtb/results.json')]
                if campaign=='seed_replication_v1':files.append(folder/'readouts/physical.json')
                for f in files:
                    data=json.loads(f.read_text());assert data['complete'];sources[str(f.relative_to(root))]=sha(f)
                    for row in data['rows']:
                        r=row['physical']['result'] if 'physical' in row else row
                        if r['success']:continue
                        detail=root/row['physical']['details'] if 'physical' in row else f.parent/'details'/r['task_id'];key=str(detail.relative_to(root))
                        xyz=detail/'input.xyz';assert sha(xyz)==r['input_xyz_sha256'];coords=np.array([[float(v) for v in line.split()[1:]] for line in xyz.read_text().splitlines()[2:]])
                        distances=np.linalg.norm(coords[:,None]-coords[None,:],axis=-1);distances[np.diag_indices(len(coords))]=np.inf
                        failures[key]=dict(original=r,source_detail=key,source_xyz_sha256=sha(xyz),graph_valid=False,natoms=len(coords),minimum_distance_A=float(distances.min()))
                        assert row.get('graph') is False and distances.min()>0
        assert len(failures)==6
        protocol=dict(frozen=True,sources=sources,failures=failures,stages=[dict(iterations=1000,broydamp=None),dict(iterations=1000,broydamp=.2)],
            timeout_seconds=300,accuracy=.1,electronic_temperature_K=300,rule='For every failed unique geometry, take the first successful stage. Increase SCC iterations, then use Broyden damping0.2 if needed. Keep coordinates, GFN2 Hamiltonian, convergence accuracy, charge, spin and electronic temperature fixed. No optimization, geometry replacement or energy-ranked solution choice.',
            scope='Numerical-completion follow-up. Original failures and their denominators remain archived. All failed graphs are invalid, so completing energies cannot change graph/force joint yields.',documentation='https://xtb-docs.readthedocs.io/en/latest/sp.html')
        proto.write_text(json.dumps(protocol,indent=2)+'\n');print('Frozen six-geometry SCC completion policy');return
    spec=json.loads(proto.read_text());assert spec['frozen'];a.out.mkdir(parents=True,exist_ok=False)
    for name,digest in spec['sources'].items():assert sha(root/name)==digest
    def evaluate(pair):
        index,(key,item)=pair;original=root/key;records=[]
        for stage,settings in enumerate(spec['stages']):
            folder=a.out/f'case{index}'/f'stage{stage}';folder.mkdir(parents=True);shutil.copy2(original/'input.xyz',folder/'input.xyz');assert sha(folder/'input.xyz')==item['source_xyz_sha256']
            control='$scc\n maxiterations='+str(settings['iterations'])+'\n temp=300\n'
            if settings['broydamp'] is not None:control+=' broydamp='+str(settings['broydamp'])+'\n'
            control+='$end\n';(folder/'solver.inp').write_text(control);old=item['original'];binary=Path(old['command'][0]);assert sha(binary)=='85da9d385d12ac9674ac311f3cdc669bb90b0423680e33317fbc631af97f78cf'
            command=[str(binary),str(folder/'input.xyz'),'--gfn','2','--acc','.1','--chrg',str(old['original_charge']),'--uhf',str(old['uhf']),'--input',str(folder/'solver.inp'),'--grad'];tick=time.perf_counter();env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            try:
                proc=subprocess.run(command,cwd=folder,env=env,capture_output=True,text=True,timeout=spec['timeout_seconds']);stdout,stderr,code=proc.stdout,proc.stderr,proc.returncode
            except subprocess.TimeoutExpired as e:
                dec=lambda s:s.decode(errors='replace') if isinstance(s,bytes) else s or '';stdout,stderr,code=dec(e.stdout),dec(e.stderr),None
            (folder/'stdout.txt').write_text(stdout);(folder/'stderr.txt').write_text(stderr);gradient=(folder/'gradient').read_text() if (folder/'gradient').exists() else ''
            result=parse_singlepoint(stdout,stderr,code,gradient,item['natoms']) if code is not None else dict(success=False,failure='timeout')
            assert sha(folder/'input.xyz')==item['source_xyz_sha256'] and '--opt' not in command
            result.update(command=command,returncode=code,seconds=time.perf_counter()-tick,stage=stage,source_detail=key,details=str(folder.relative_to(root)),input_xyz_sha256=sha(folder/'input.xyz'),stdout_sha256=sha(folder/'stdout.txt'),stderr_sha256=sha(folder/'stderr.txt'),gradient_sha256=sha(folder/'gradient') if (folder/'gradient').exists() else None,solver_sha256=sha(folder/'solver.inp'),original_charge=old['original_charge'],original_spin_multiplicity=old['original_spin_multiplicity'])
            records.append(result)
            if result['success']:break
        return dict(source_detail=key,attempts=records,successful=records[-1]['success'])
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(evaluate,enumerate(spec['failures'].items())))
    result=dict(complete=True,protocol_sha256=sha(proto),rows=rows,new_gfn2_attempts=sum(len(r['attempts']) for r in rows),successful=sum(r['successful'] for r in rows),requested=len(rows),geometry_optimization=False)
    (a.out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(successful=result['successful'],requested=len(rows),new_gfn2_attempts=result['new_gfn2_attempts'])))

if __name__=='__main__':main()
