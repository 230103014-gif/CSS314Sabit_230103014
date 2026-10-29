import sys
sys.dont_write_bytecode = True
import argparse
import ctypes
import json
import math
import multiprocessing as mp
import statistics
import time

from salt import limit_numpy_threads
limit_numpy_threads()
import numpy as np
from salt import output_dir, read_params, write_json, write_tsv, numeric_digest


def stencil_step(src, dst, lo, hi):
    target = dst[lo:hi, 1:-1]
    np.add(src[lo:hi, 1:-1], src[lo-1:hi-1, 1:-1], out=target)
    np.add(target, src[lo+1:hi+1, 1:-1], out=target)
    np.add(target, src[lo:hi, :-2], out=target)
    np.add(target, src[lo:hi, 2:], out=target)
    np.multiply(target, 0.2, out=target)


def initial_grid(n, seed):
    a = np.random.default_rng(seed).random((n, n))
    a[0, :] = a[-1, :] = 0
    a[:, 0] = a[:, -1] = 0
    return a


def reference_stencil(initial, steps):
    a, b = initial.copy(), np.zeros_like(initial)
    for _ in range(steps):
        stencil_step(a, b, 1, len(a)-1)
        a, b = b, a
    return a


def stencil_worker(raw_a, raw_b, n, lo, hi, steps, barrier, ready, start, done):
    try:
        a = np.frombuffer(raw_a, dtype=np.float64).reshape(n, n)
        b = np.frombuffer(raw_b, dtype=np.float64).reshape(n, n)
        ready.put(lo)
        if not start.wait(120):
            raise TimeoutError('start event')
        begin = time.perf_counter_ns()
        for _ in range(steps):
            stencil_step(a, b, lo, hi)
            barrier.wait(timeout=120)
            a, b = b, a
        done.put({'elapsed_ns': time.perf_counter_ns()-begin})
    except BaseException as exc:
        barrier.abort()
        done.put({'error': repr(exc)})
        raise


def trial(initial, p, steps):
    n = len(initial)
    ctx = mp.get_context('spawn')
    raw_a = ctx.RawArray(ctypes.c_double, n*n)
    raw_b = ctx.RawArray(ctypes.c_double, n*n)
    a = np.frombuffer(raw_a, dtype=np.float64).reshape(n, n)
    a[:] = initial
    ready, done = ctx.Queue(), ctx.Queue()
    start, barrier = ctx.Event(), ctx.Barrier(p)
    processes = []
    for i in range(p):
        lo, hi = 1+(n-2)*i//p, 1+(n-2)*(i+1)//p
        processes.append(ctx.Process(target=stencil_worker,
            args=(raw_a, raw_b, n, lo, hi, steps, barrier, ready, start, done)))
    try:
        for proc in processes:
            proc.start()
        for _ in processes:
            ready.get(timeout=120)
        begin = time.perf_counter_ns()
        start.set()
        records = [done.get(timeout=300) for _ in processes]
        elapsed = time.perf_counter_ns()-begin
        for proc in processes:
            proc.join(timeout=30)
            if proc.exitcode != 0:
                raise RuntimeError('Stencil worker failed: ' + str(proc.exitcode))
        if any('error' in r for r in records):
            raise RuntimeError(str(records))
        output = np.frombuffer(raw_a if steps%2 == 0 else raw_b, dtype=np.float64).reshape(n,n).copy()
        return elapsed, output
    finally:
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
                proc.join()
        ready.close()
        done.close()


def run(out, smoke=False, peak_bandwidth=None, peak_gflops=None):
    params = read_params(out)
    profile = json.loads((out/'host_profile.json').read_text())
    physical = profile.get('physical_cores')
    max_p = 2 if smoke else (physical or profile.get('logical_threads_fallback') or 1)
    ps = [1]
    while ps[-1]*2 <= max_p:
        ps.append(ps[-1]*2)
    if max_p not in ps:
        ps.append(max_p)
    base_n = 64 if smoke else params['N']
    steps = 3 if smoke else 100
    repeats = 1 if smoke else 3
    rows = []
    for study in ('strong', 'weak'):
        t1 = None
        for p in ps:
            n = base_n if study == 'strong' else math.ceil(base_n*math.sqrt(p))
            initial = initial_grid(n, params['seed'])
            expected = reference_stencil(initial, steps)
            if not smoke:
                trial(initial, p, steps)
            times = []
            for _ in range(repeats):
                elapsed, actual = trial(initial, p, steps)
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
                times.append(elapsed)
            median = int(statistics.median(times))
            if p == 1:
                t1 = median
            updates = (n-2)**2*steps
            base_updates = (base_n-2)**2*steps
            ratio = updates/base_updates
            speedup = t1/median
            scaled = ratio*speedup
            s = (1/speedup-1/p)/(1-1/p) if p > 1 and study == 'strong' else ''
            key = f'{study}_p{p}'
            rows.append({'study': study, 'processes': p, 'N': n, 'steps': steps,
                         'cell_updates': updates, 'work_ratio': ratio, 'runs_ns': json.dumps(times),
                         'median_ns': median, 'speedup_T1_over_TP': speedup,
                         'efficiency_percent': (speedup if study=='strong' else scaled)/p*100,
                         'scaled_speedup_work_ratio_T1_over_TP': scaled,
                         'weak_pdf_literal': p/ratio*speedup if study=='weak' else '',
                         'gustafson_alpha': (p-scaled)/(p-1) if study=='weak' and p>1 else '',
                         'amdahl_s_point': s, 'G_updates_per_s': updates/median,
                         'GFLOP_per_s': 5*updates/median,
                         'checksum_sum': float(actual.sum()),
                         'digest': numeric_digest(actual, params['student_id']), 'array_key': key,
                         'smoke': smoke})
            write_tsv(out/'scaling_metrics.tsv', rows)
            print('stencil', study, p, n, 'seconds', round(median/1e9, 3), flush=True)
    strong = [r for r in rows if r['study']=='strong' and r['processes']>1]
    x = np.array([1-1/r['processes'] for r in strong])
    y = np.array([1/r['speedup_T1_over_TP']-1/r['processes'] for r in strong])
    raw_s = float(x@y/(x@x)) if len(x) else None
    fitted_s = min(1.0, max(0.0, raw_s)) if raw_s is not None else None
    fit = {'unconstrained_s': raw_s, 'constrained_s': fitted_s,
           'Smax': 1/fitted_s if fitted_s and fitted_s>0 else None,
           'Smax_note': 'null means no finite ceiling estimated' if fitted_s == 0 else None,
           'method': 'least squares on T_P/T_1 = 1/P + s*(1-1/P); s constrained to [0,1]',
           'model_warning': 'Amdahl fit is descriptive; heterogeneous cores, barriers and memory traffic violate ideal assumptions.',
           'rmse_normalized_runtime': float(np.sqrt(np.mean((y-fitted_s*x)**2))) if len(x) else None,
           'steps': steps, 'workers': ps, 'physical_core_count_verified': physical is not None,
           'timing_scope': 'spawn, allocation, initialization and reference excluded; start signal, 100 update steps with per-step barriers, and completion notification included',
           'warmup_trials_per_configuration': 0 if smoke else 1, 'recorded_trials': repeats}
    best = max((r for r in rows if r['study']=='strong'), key=lambda r:r['GFLOP_per_s'])
    ai_ideal, ai_no_reuse = 5/24, 5/48
    regime = 'conditional memory-bandwidth-bound: low AI, hardware ridge point unavailable'
    if peak_bandwidth and peak_gflops:
        regime = 'Memory-Bandwidth Bound (theoretical model)' if ai_ideal < peak_gflops/peak_bandwidth else 'Compute Bound (theoretical model)'
    write_json(out/'roofline_summary.json', {
        'scaling_fit': fit,
        'kernel': 'five-point Jacobi: (center+north+south+west+east)*0.2, float64; zero Dirichlet boundary',
        'FLOPs_per_update': 5, 'minimal_DRAM_bytes_per_update_with_write_allocate': 24,
        'no_spatial_reuse_bytes_per_update': 48, 'AI_ideal_FLOPs_per_byte': ai_ideal,
        'AI_no_reuse_FLOPs_per_byte': ai_no_reuse,
        'traffic_assumptions': '24 B assumes each source loaded once, one 8 B destination RFO plus 8 B writeback; 48 B assumes five independent 8 B source loads plus one 8 B store. These are alternative models, not measured DRAM traffic. NumPy uses multiple passes and can add traffic.',
        'two_grid_working_set_bytes': 2*base_n**2*8,
        'cache_sizes': {k:profile.get(k) for k in ('l1_data_cache_bytes','l2_cache_bytes','l3_cache_bytes')},
        'cache_residency_warning': 'If grids fit shared cache, DRAM Roofline is insufficient; use a cache-level Roofline. Barriers, Python/NumPy dispatch and scheduler overhead can also dominate.',
        'peak_memory_bandwidth_GB_per_s': peak_bandwidth, 'peak_compute_GFLOP_per_s': peak_gflops,
        'ridge_point_FLOPs_per_byte': peak_gflops/peak_bandwidth if peak_bandwidth and peak_gflops else None,
        'classification': regime, 'best_measured_GFLOP_per_s': best['GFLOP_per_s'],
        'bandwidth_roof_GFLOP_per_s_ideal_AI': ai_ideal*peak_bandwidth if peak_bandwidth else None,
        'evidence_status': 'analytical traffic estimate only; no hardware counter proof',
        'weak_scaling_note': 'PDF expression P*Work(1)*T1/(Work(P)*TP) approaches weak efficiency T1/TP when Work(P)=P*Work(1). Conventional scaled speedup is Work(P)/Work(1)*T1/TP. Both are preserved.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--peak-memory-bandwidth-gbps', type=float)
    parser.add_argument('--peak-compute-gflops', type=float)
    a = parser.parse_args()
    run(output_dir(a.results_dir), a.smoke, a.peak_memory_bandwidth_gbps, a.peak_compute_gflops)
