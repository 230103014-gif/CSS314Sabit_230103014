"""M1 dense loop-order benchmark; embedded C kernels are compiled via ctypes."""
import sys
sys.dont_write_bytecode = True
import ctypes
import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from salt import ROOT


NATIVE_SOURCE = '#include <stdint.h>\n#include <stddef.h>\n\ndouble stride_sum(const double *a, size_t n, size_t stride) {\n    double total = 0.0;\n    for (size_t i = 0; i < n; i += stride) total += a[i];\n    return total;\n}\n\nvoid dense_ijk(const double *a, const double *b, double *c, size_t ld, size_t m) {\n    for (size_t i = 0; i < m; ++i)\n        for (size_t j = 0; j < m; ++j) {\n            double acc = 0.0;\n            for (size_t k = 0; k < m; ++k) acc += a[i*ld+k] * b[k*ld+j];\n            c[i*ld+j] = acc;\n        }\n}\n\nvoid dense_ikj(const double *a, const double *b, double *c, size_t ld, size_t m) {\n    for (size_t i = 0; i < m; ++i)\n        for (size_t j = 0; j < m; ++j) c[i*ld+j] = 0.0;\n    for (size_t i = 0; i < m; ++i)\n        for (size_t k = 0; k < m; ++k) {\n            double aik = a[i*ld+k];\n            for (size_t j = 0; j < m; ++j) c[i*ld+j] += aik * b[k*ld+j];\n        }\n}\n'
_NATIVE_BUILD = tempfile.TemporaryDirectory(prefix="cs482-kernels-")


def load_native():
    source = Path(_NATIVE_BUILD.name)/"native_kernels.c"
    source.write_text(NATIVE_SOURCE)
    compiler = shutil.which('clang') or shutil.which('cc')
    if not compiler:
        raise RuntimeError('A C compiler is required for the ctypes dense/stride kernels. Install Xcode Command Line Tools on macOS.')
    flags = ['-O2', '-fno-fast-math', '-ffp-contract=off']
    if 'clang' in compiler or platform.system() == 'Darwin':
        flags += ['-fno-vectorize', '-fno-slp-vectorize']
    else:
        flags += ['-fno-tree-vectorize', '-fno-loop-interchange']
    build = Path(_NATIVE_BUILD.name)
    build.mkdir(exist_ok=True)
    target = build/('kernels.dylib' if platform.system() == 'Darwin' else 'kernels.so')
    fingerprint = hashlib.sha256(source.read_bytes() + json.dumps([compiler, flags]).encode()).hexdigest()
    stamp = build/'fingerprint.txt'
    if not target.exists() or not stamp.exists() or stamp.read_text() != fingerprint:
        subprocess.run([compiler, *flags, '-shared', '-fPIC', str(source), '-o', str(target)], check=True)
        stamp.write_text(fingerprint)
    lib = ctypes.CDLL(str(target))
    ptr = ctypes.POINTER(ctypes.c_double)
    lib.stride_sum.argtypes = [ptr, ctypes.c_size_t, ctypes.c_size_t]
    lib.stride_sum.restype = ctypes.c_double
    for name in ('dense_ijk', 'dense_ikj'):
        f = getattr(lib, name)
        f.argtypes = [ptr, ptr, ptr, ctypes.c_size_t, ctypes.c_size_t]
        f.restype = None
    return lib, {'compiler': compiler, 'flags': flags, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                 'backend': 'single-threaded C loops via standard-library ctypes; no BLAS'}


def pointer(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


import argparse
import numpy as np
from salt import measured, output_dir, read_params, write_json, numeric_digest


def run(out, smoke=False):
    params = read_params(out)
    n = 64 if smoke else params['N']
    m = n//2
    rng = np.random.default_rng(params['seed'])
    a, b = rng.random((n, n)), rng.random((n, n))
    c0, c1 = np.zeros((n, n)), np.zeros((n, n))
    lib, backend = load_native()
    def call(name, c):
        getattr(lib, name)(pointer(a), pointer(b), pointer(c), n, m)
    _, naive, t0 = measured(lambda: call('dense_ijk', c0), repeats=1 if smoke else 3)
    _, optimized, t1 = measured(lambda: call('dense_ikj', c1), repeats=1 if smoke else 3)
    baseline, interchanged = c0[:m, :m].copy(), c1[:m, :m].copy()
    np.testing.assert_allclose(baseline, interchanged, rtol=0, atol=1e-12)
    reference = a[:m, :m] @ b[:m, :m]
    np.testing.assert_allclose(baseline, reference, rtol=0, atol=1e-12)
    write_json(out/'loop_interchange.json', {
        'N': n, 'M': m, 'leading_dimension': n, 'seed': params['seed'], 'smoke': smoke,
        'algorithm': 'C=A[:M,:M] @ B[:M,:M]; parent arrays retain salted leading dimension N',
        'backend': backend, 'warmups': 1, 'naive_runs_ns': naive,
        'interchanged_runs_ns': optimized, 'naive_median_ns': t0,
        'interchanged_median_ns': t1, 'speedup': t0/t1,
        'max_absolute_error': float(np.max(np.abs(baseline-interchanged))),
        'baseline_digest': numeric_digest(baseline, params['student_id']),
        'interchanged_digest': numeric_digest(interchanged, params['student_id']),
        'numpy_reference_max_absolute_error': float(np.max(np.abs(baseline-reference)))})
    print('dense speedup:', t0/t1, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    run(output_dir(a.results_dir), a.smoke)
