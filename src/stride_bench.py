import sys
sys.dont_write_bytecode = True
import argparse
import json
import numpy as np
from salt import measured, output_dir, read_params, write_tsv, write_json
from dense_kernel import load_native, pointer


def run(out, smoke=False):
    params = read_params(out)
    n = 262144 if smoke else params['buffer_elements']
    data = np.ones(n, dtype=np.float64)
    lib, backend = load_native()
    rows = []
    for stride in (1, 2, 4, 8, 16, 32, 64, 128):
        total, times, median = measured(lambda: lib.stride_sum(pointer(data), n, stride),
                                        repeats=1 if smoke else 5, warmups=1 if smoke else 5)
        touched = (n + stride - 1)//stride
        assert total == touched
        rows.append({'stride': stride, 'buffer_elements': n, 'elements_touched': touched,
                     'warmups': 1 if smoke else 5, 'runs_ns': json.dumps(times),
                     'median_ns': median, 'ns_per_element': median/touched,
                     'useful_GB_per_s': touched*8/median, 'sum': total})
        print('stride', stride, 'ns/element', round(median/touched, 3), flush=True)
    write_tsv(out/'stride_results.tsv', rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    run(output_dir(a.results_dir), a.smoke)
