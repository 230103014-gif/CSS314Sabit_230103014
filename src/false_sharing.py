import sys
sys.dont_write_bytecode = True
import argparse
import ctypes
import json
import multiprocessing as mp
import queue
import statistics
import time

from salt import output_dir, read_params, write_json, write_tsv


def counter_worker(arr, offset, step, worker_id, iterations, mode, ready, start, done, final_lock):
    try:
        index = offset + worker_id*step
        ready.put(worker_id)
        if not start.wait(120):
            raise TimeoutError('start event')
        begin = time.perf_counter_ns()
        if mode == 'privatized':
            value = 0
            for _ in range(iterations):
                value += 1
            with final_lock:
                arr[index] = value
        else:
            for _ in range(iterations):
                arr[index] += 1
        done.put({'worker': worker_id, 'elapsed_ns': time.perf_counter_ns()-begin})
    except BaseException as exc:
        done.put({'error': repr(exc)})
        raise


def trial(p, k, step, mode, alignment):
    ctx = mp.get_context('spawn')
    arr = ctx.RawArray(ctypes.c_int64, p*step + alignment//8)
    address = ctypes.addressof(arr)
    offset = ((-address) % alignment)//8
    ready, done = ctx.Queue(), ctx.Queue()
    start, final_lock = ctx.Event(), ctx.Lock()
    processes = [ctx.Process(target=counter_worker, args=(arr, offset, step, i, k, mode,
                 ready, start, done, final_lock)) for i in range(p)]
    try:
        for proc in processes:
            proc.start()
        for _ in processes:
            ready.get(timeout=120)
        begin = time.perf_counter_ns()
        start.set()
        records = [done.get(timeout=600) for _ in processes]
        elapsed = time.perf_counter_ns()-begin
        for proc in processes:
            proc.join(timeout=30)
            if proc.exitcode != 0:
                raise RuntimeError('Counter worker failed: ' + str(proc.exitcode))
        if any('error' in item for item in records):
            raise RuntimeError(str(records))
        counters = [int(arr[offset+i*step]) for i in range(p)]
        if counters != [k]*p:
            raise AssertionError(counters)
        return elapsed, counters, records, address + offset*8
    finally:
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
                proc.join()
        ready.close()
        done.close()


def run(out, smoke=False):
    read_params(out)
    profile = json.loads((out/'host_profile.json').read_text())
    line = profile.get('cache_line_bytes')
    alignment = max(128, line or 128)
    configurations = [('unpadded', 1), ('padded64', 8), ('privatized', alignment//8)]
    if line and line > 64:
        configurations.insert(2, ('padded_native', (line+7)//8))
    rows = []
    k = 10000 if smoke else 5_000_000
    for mode, step in configurations:
        baseline = None
        for p in ((1, 2) if smoke else (1, 2, 4, 8)):
            elapsed, counts, workers, address = trial(p, k, step, mode, alignment)
            if p == 1:
                baseline = elapsed
            speedup = baseline/elapsed
            normalized_speedup = p*speedup
            e = (p/speedup-1)/(p-1) if p > 1 else ''
            e_normalized = (p/normalized_speedup-1)/(p-1) if p > 1 else ''
            rows.append({'mode': mode, 'processes': p, 'iterations_per_worker': k,
                         'counter_spacing_bytes': step*8, 'detected_cache_line_bytes': line or '',
                         'first_counter_address': address, 'alignment_bytes': alignment,
                         'elapsed_ns': elapsed, 'runtime_ratio_speedup': speedup,
                         'karp_flatt_pdf_literal': e,
                         'work_normalized_speedup': normalized_speedup,
                         'karp_flatt_work_normalized': e_normalized,
                         'total_counts': sum(counts), 'expected_counts': p*k,
                         'counters': json.dumps(counts), 'workers': json.dumps(workers),
                         'smoke': smoke})
            write_tsv(out/'false_sharing_karp_flatt.tsv', rows)
            print('counters', mode, p, 'time_s', round(elapsed/1e9, 3), flush=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir')
    parser.add_argument('--smoke', action='store_true')
    a = parser.parse_args()
    run(output_dir(a.results_dir), a.smoke)
