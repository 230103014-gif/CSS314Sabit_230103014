"""Deterministic Student ID salting and shared timing, digest and file helpers."""
import sys
sys.dont_write_bytecode = True
import hashlib


def get_student_params(student_id: str):
    if not student_id.isdigit():
        raise ValueError('Student ID must contain digits; leading zeros are preserved.')
    h = int(hashlib.sha256(student_id.encode('utf-8')).hexdigest()[:8], 16)
    return {'student_id': student_id, 'N': 1024 + ((h % 9) * 64),
            'buffer_elements': 16_777_216, 'seed': h % 100_007}


import csv
import json
import os
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]


def output_dir(path=None):
    result = Path(path) if path else ROOT / 'results'
    result.mkdir(parents=True, exist_ok=True)
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_tsv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def read_params(directory):
    return json.loads((directory / 'salted_params.json').read_text())


def numeric_digest(array, student_id):
    import numpy as np
    a = np.asarray(array)
    header = {'student_id': student_id, 'shape': list(a.shape), 'round_decimals': 6}
    h = hashlib.sha256(json.dumps(header, sort_keys=True).encode())
    if np.issubdtype(a.dtype, np.integer):
        h.update(a.astype('<i8').tobytes(order='C'))
    else:
        rounded = np.round(a.astype(np.float64), 6)
        rounded[rounded == 0] = 0.0
        h.update(rounded.astype('<f8').tobytes(order='C'))
    return h.hexdigest()


def measured(fn, repeats=3, warmups=1):
    for _ in range(warmups):
        fn()
    runs = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter_ns()
        value = fn()
        runs.append(time.perf_counter_ns() - start)
    return value, runs, int(statistics.median(runs))


def limit_numpy_threads():
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
