"""Verify recorded data and regenerate numerical outputs without storing large arrays.

Python 3.10+, NumPy and clang/cc are required. Run: python3 verify_outputs.py
This verifies current source/results; it does not authenticate original timings.
Use --check-git to also require the five actual milestone tags.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))
from salt import numeric_digest, output_dir, write_json, get_student_params, limit_numpy_threads
limit_numpy_threads()
import numpy as np
from dense_kernel import load_native, pointer
from scaling_study import initial_grid, trial as stencil_trial


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def tsv(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def independent_stencil(n, seed, steps):
    a = np.random.default_rng(seed).random((n,n))
    a[0,:] = a[-1,:] = 0
    a[:,0] = a[:,-1] = 0
    for _ in range(steps):
        b = np.zeros_like(a)
        b[1:-1,1:-1] = ((((a[1:-1,1:-1] + a[:-2,1:-1]) + a[2:,1:-1])
                         + a[1:-1,:-2]) + a[1:-1,2:])*0.2
        a = b
    return a


def verify(out, smoke=False, check_git=False, allow_incomplete_telemetry=False):
    receipt = out/'verification_receipt.sig'
    params = json.loads((out/'salted_params.json').read_text())
    require(params == get_student_params(params['student_id']), 'Salted parameters do not match the full Student ID')
    profile = json.loads((out/'host_profile.json').read_text())
    if not smoke and not allow_incomplete_telemetry:
        require(profile['required_telemetry_complete'], 'Required hardware telemetry is unavailable. Run outside the restricted environment.')
    print('[PASS] M0 salting; telemetry complete:',profile['required_telemetry_complete'])
    strides = tsv(out/'stride_results.tsv')
    require([int(r['stride']) for r in strides] == [1,2,4,8,16,32,64,128], 'Missing stride configurations')
    for r in strides:
        n, s = int(r['buffer_elements']), int(r['stride'])
        count = (n+s-1)//s
        require(int(r['elements_touched']) == count and float(r['sum']) == count, 'Incorrect stride sum')
        runs = json.loads(r['runs_ns'])
        require(all(t>0 for t in runs), 'Nonpositive stride runtime')
        require(len(runs) == (1 if smoke else 5), 'Wrong recorded stride run count')
        require(int(r['warmups']) == (1 if smoke else 5), 'Wrong stride warmup count')
        require(int(r['median_ns']) == int(statistics.median(runs)), 'Stride median mismatch')
        require(math.isclose(float(r['ns_per_element']), int(r['median_ns'])/count), 'Stride normalization mismatch')
        require(n == (262144 if smoke else params['buffer_elements']), 'Wrong stride buffer size')
    loops = json.loads((out/'loop_interchange.json').read_text())
    n = 64 if smoke else params['N']
    m = n//2
    require(loops['N'] == n and loops['M'] == m and loops['leading_dimension'] == n, 'Wrong dense dimensions')
    rng = np.random.default_rng(params['seed'])
    a, b = rng.random((n,n)), rng.random((n,n))
    expected = a[:m,:m] @ b[:m,:m]
    native, backend = load_native()
    require(backend['source_sha256'] == loops['backend']['source_sha256'], 'Native kernel source differs from measured implementation')
    baseline, interchanged = np.zeros((n,n)), np.zeros((n,n))
    native.dense_ijk(pointer(a),pointer(b),pointer(baseline),n,m)
    native.dense_ikj(pointer(a),pointer(b),pointer(interchanged),n,m)
    for actual, digest_key in [(baseline[:m,:m],'baseline_digest'),(interchanged[:m,:m],'interchanged_digest')]:
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
        require(numeric_digest(actual,params['student_id']) == loops[digest_key], 'Dense digest differs from the originally recorded output')
    np.testing.assert_allclose(baseline[:m,:m], interchanged[:m,:m], rtol=0, atol=1e-12)
    require(0 <= loops['max_absolute_error'] <= 1e-12, 'Recorded matrix error outside tolerance')
    for kind in ('naive','interchanged'):
        runs = loops[kind+'_runs_ns']
        require(len(runs) == (1 if smoke else 3) and min(runs)>0, 'Wrong dense trials')
        require(loops[kind+'_median_ns'] == int(statistics.median(runs)), 'Dense median mismatch')
    require(math.isclose(loops['speedup'], loops['naive_median_ns']/loops['interchanged_median_ns']), 'Dense speedup mismatch')
    print('[PASS] M1 stride sums, matrix reference, tolerance and salted digests')
    counters = tsv(out/'false_sharing_karp_flatt.tsv')
    modes = {'unpadded','padded64','privatized'}
    if profile.get('cache_line_bytes') and profile['cache_line_bytes']>64:
        modes.add('padded_native')
    ps = [1,2] if smoke else [1,2,4,8]
    require(len(counters) == len(modes)*len(ps), 'Wrong number of counter rows')
    require({(r['mode'],int(r['processes'])) for r in counters} == {(mode,p) for mode in modes for p in ps}, 'Missing counter configurations')
    for mode in modes:
        group = [r for r in counters if r['mode']==mode]
        base = next(int(r['elapsed_ns']) for r in group if int(r['processes'])==1)
        for r in group:
            p, k, elapsed = int(r['processes']), int(r['iterations_per_worker']), int(r['elapsed_ns'])
            require(k == (10000 if smoke else 5_000_000), 'Wrong counter iterations')
            require(json.loads(r['counters']) == [k]*p, 'Per-worker counters incorrect')
            require(int(r['total_counts']) == int(r['expected_counts']) == p*k, 'Counter total incorrect')
            require(elapsed>0, 'Counter time invalid')
            s = base/elapsed
            require(math.isclose(float(r['runtime_ratio_speedup']),s), 'Counter speedup incorrect')
            if p>1:
                require(math.isclose(float(r['karp_flatt_pdf_literal']),(p/s-1)/(p-1)), 'Literal Karp-Flatt incorrect')
                require(math.isclose(float(r['karp_flatt_work_normalized']),(1/s-1)/(p-1),abs_tol=1e-12), 'Normalized Karp-Flatt incorrect')
            require(int(r['first_counter_address']) % int(r['alignment_bytes']) == 0, 'Counter base not aligned')
    print('[PASS] M2 all per-worker counts and Karp-Flatt arithmetic')
    scaling = tsv(out/'scaling_metrics.tsv')
    roof = json.loads((out/'roofline_summary.json').read_text())
    fit = roof['scaling_fit']
    require({(r['study'],int(r['processes'])) for r in scaling} == {(study,p) for study in ('strong','weak') for p in fit['workers']}, 'Missing scaling cases')
    base_n = 64 if smoke else params['N']
    for study in ('strong','weak'):
        group = [r for r in scaling if r['study']==study]
        t1 = next(int(r['median_ns']) for r in group if int(r['processes'])==1)
        for r in group:
            p,n,steps = int(r['processes']),int(r['N']),int(r['steps'])
            require(n == (base_n if study=='strong' else math.ceil(base_n*math.sqrt(p))), 'Wrong scaling dimension')
            require(steps == (3 if smoke else 100), 'Wrong stencil steps')
            runs = json.loads(r['runs_ns'])
            require(len(runs)==(1 if smoke else 3) and min(runs)>0, 'Wrong scaling trials')
            require(int(r['median_ns'])==int(statistics.median(runs)), 'Scaling median mismatch')
            expected = independent_stencil(n,params['seed'],steps)
            _, actual = stencil_trial(initial_grid(n,params['seed']),p,steps)
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
            require(numeric_digest(actual,params['student_id'])==r['digest'], 'Regenerated stencil differs from recorded salted digest')
            require(math.isclose(float(actual.sum()),float(r['checksum_sum']),rel_tol=1e-12,abs_tol=1e-12), 'Recorded stencil reduction differs')
            speedup = t1/int(r['median_ns'])
            work = (n-2)**2*steps
            ratio = (n-2)**2/(base_n-2)**2
            require(int(r['cell_updates'])==work, 'Stencil work count mismatch')
            require(math.isclose(float(r['speedup_T1_over_TP']),speedup), 'Scaling speedup mismatch')
            require(math.isclose(float(r['work_ratio']),ratio), 'Work ratio mismatch')
            require(math.isclose(float(r['scaled_speedup_work_ratio_T1_over_TP']),ratio*speedup), 'Scaled speedup mismatch')
            efficiency = (speedup if study=='strong' else ratio*speedup)/p*100
            require(math.isclose(float(r['efficiency_percent']),efficiency), 'Efficiency mismatch')
            require(math.isclose(float(r['GFLOP_per_s']),5*work/int(r['median_ns'])), 'Stencil GFLOP/s mismatch')
            if study=='strong' and p>1:
                expected_s=(1/speedup-1/p)/(1-1/p)
                require(math.isclose(float(r['amdahl_s_point']),expected_s,abs_tol=1e-12), 'Amdahl point estimate mismatch')
            if study=='weak':
                require(math.isclose(float(r['weak_pdf_literal']),p/ratio*speedup), 'PDF weak scaling expression mismatch')
                if p>1:
                    require(math.isclose(float(r['gustafson_alpha']),(p-ratio*speedup)/(p-1),abs_tol=1e-12), 'Gustafson parameter mismatch')
            print('[PASS] Regenerated',study,'P='+str(p),'matches recorded digest',flush=True)
    strong=[r for r in scaling if r['study']=='strong' and int(r['processes'])>1]
    x=np.array([1-1/int(r['processes']) for r in strong])
    y=np.array([1/float(r['speedup_T1_over_TP'])-1/int(r['processes']) for r in strong])
    raw_s=float(x@y/(x@x)) if len(x) else None
    if raw_s is not None:
        require(math.isclose(fit['unconstrained_s'],raw_s,abs_tol=1e-12), 'Fitted Amdahl fraction mismatch')
        fitted=min(1,max(0,raw_s))
        require(math.isclose(fit['constrained_s'],fitted,abs_tol=1e-12), 'Constrained Amdahl fit mismatch')
        if fitted>0:
            require(math.isclose(fit['Smax'],1/fitted), 'Amdahl ceiling mismatch')
        require(math.isclose(fit['rmse_normalized_runtime'],float(np.sqrt(np.mean((y-fitted*x)**2)))), 'Amdahl fit residual mismatch')
    roof = json.loads((out/'roofline_summary.json').read_text())
    require(roof['FLOPs_per_update']==5 and math.isclose(roof['AI_ideal_FLOPs_per_byte'],5/24), 'Arithmetic intensity mismatch')
    if not smoke and not allow_incomplete_telemetry:
        require(fit['physical_core_count_verified'], 'Scaling core count was a fallback, not verified physical cores')
    print('[PASS] M3 every stencil array vs independent serial reference; scaling arithmetic')
    observed = {}
    stride1 = float(strides[0]['ns_per_element'])
    stride8 = float(next(r['ns_per_element'] for r in strides if int(r['stride'])==8))
    observed['stride8_to_stride1_latency_ratio'] = stride8/stride1
    if not smoke:
        def elapsed(mode,p):
            return int(next(r['elapsed_ns'] for r in counters if r['mode']==mode and int(r['processes'])==p))
        chosen = 'padded_native' if 'padded_native' in modes else 'padded64'
        observed['unpadded_T8_over_T1'] = elapsed('unpadded',8)/elapsed('unpadded',1)
        observed['unpadded_over_padded_T8'] = elapsed('unpadded',8)/elapsed(chosen,8)
    warnings = []
    if not profile['required_telemetry_complete']:
        warnings.append('Required hardware telemetry incomplete; not a complete empirical submission.')
    if not fit['physical_core_count_verified']:
        warnings.append('Scaling used logical-thread fallback; physical-core worker limit is unverified.')
    if not roof.get('peak_memory_bandwidth_GB_per_s') or not roof.get('peak_compute_GFLOP_per_s'):
        warnings.append('Theoretical CPU peaks not supplied: Roofline regime remains conditional.')
    if observed['stride8_to_stride1_latency_ratio'] <= 1:
        warnings.append('No stride-8 latency increase was observed; expected cliff not demonstrated.')
    if not smoke and observed['unpadded_over_padded_T8'] <= 1:
        warnings.append('Padded recovery was not observed at P=8; false-sharing recovery criterion not demonstrated.')
    if smoke:
        warnings.append('Reduced-size smoke test only; not full practicum data.')
    tags = []
    if check_git:
        repo = Path(__file__).resolve().parent
        actual_root = Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=repo,text=True).strip()).resolve()
        require(actual_root==repo, 'Git repository root must be the submission directory')
        tags = subprocess.check_output(['git','tag','--list'],cwd=repo,text=True).splitlines()
        require({'M0-ENV','M1-LOCALITY','M2-SHARING','M3-SCALING','M4-FINAL'}.issubset(tags), 'Missing required milestone tags')
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file() and p.name!='verification_receipt.sig'}
    payload = {'format': 'local-verifier-sha256-v1', 'student_id':params['student_id'],
               'verified_at_utc':datetime.now(timezone.utc).isoformat(), 'smoke':smoke,
               'numerical_checks_passed':True, 'observations':observed,'warnings':warnings,
               'implemented_evidence_checks_passed':not smoke and not warnings,
               'git_history_checked':check_git,
               'submission_requirements_all_confirmed':not smoke and not warnings and check_git,
               'verification_mode':'regenerate actual kernels and independent references; compare originally recorded salted digests; original timings preserved',
               'file_sha256':hashes,'milestone_tags_checked':tags,
               'source_sha256':{str(p.relative_to(Path(__file__).resolve().parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((Path(__file__).resolve().parent/'src').glob('*.py'))},
               'authenticity_note':'Locally generated integrity digest, not a digital signature or instructor-issued verification. Does not authenticate authorship or timing.'}
    payload['receipt_sha256'] = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write_json(receipt,payload)
    print('Numerical checks passed. Implemented evidence checks passed:',payload['implemented_evidence_checks_passed'])
    if not check_git:
        print('[NOTE] Git milestone history not checked; numerical PASS is not confirmation of all submission requirements.')
    for warning in warnings:
        print('[NOTE]',warning)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results-dir')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--check-git',action='store_true')
    parser.add_argument('--allow-incomplete-telemetry',action='store_true',help='Verify numerical data only; receipt explicitly remains incomplete')
    args=parser.parse_args()
    try:
        verify(output_dir(args.results_dir),args.smoke,args.check_git,args.allow_incomplete_telemetry)
    except (AssertionError, OSError, KeyError, ValueError, subprocess.CalledProcessError) as exc:
        print('[FAIL]',exc,file=sys.stderr)
        raise SystemExit(1)
