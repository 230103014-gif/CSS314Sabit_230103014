import sys
sys.dont_write_bytecode = True
import argparse
from datetime import datetime, timezone
import json
import os
import platform
import re
import subprocess
import sys

from salt import output_dir, write_json
from salt import get_student_params


def profile():
    errors = {}
    raw = {}
    def sysctl(key):
        r = subprocess.run(['/usr/sbin/sysctl', '-n', key], capture_output=True, text=True)
        if r.returncode:
            errors[key] = r.stderr.strip()
            return None
        raw[key] = r.stdout.strip()
        if not raw[key]:
            return None
        try:
            return int(raw[key])
        except ValueError:
            return raw[key]
    p = {'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
         'system': platform.system(), 'kernel': platform.release(),
         'machine': platform.machine(), 'python': sys.version,
         'python_implementation': platform.python_implementation(),
         'logical_threads_fallback': os.cpu_count(), 'telemetry_errors': errors,
         'cpu_model': None, 'microarchitecture': None, 'base_frequency_hz': None,
         'physical_cores': None, 'logical_threads': None,
         'l1_data_cache_bytes': None, 'l2_cache_bytes': None,
         'l3_cache_bytes': None, 'cache_line_bytes': None,
         'counter_hardware_events': 'not collected; no PMU access assumed'}
    if platform.system() == 'Darwin':
        keys = {'cpu_model': 'machdep.cpu.brand_string', 'hardware_model': 'hw.model',
                'physical_cores': 'hw.physicalcpu', 'logical_threads': 'hw.logicalcpu',
                'base_frequency_hz': 'hw.cpufrequency', 'cache_line_bytes': 'hw.cachelinesize',
                'l1_data_cache_bytes': 'hw.l1dcachesize', 'l2_cache_bytes': 'hw.l2cachesize',
                'l3_cache_bytes': 'hw.l3cachesize'}
        for name, key in keys.items():
            p[name] = sysctl(key)
        levels = []
        for i in range(2):
            level = {name: sysctl(f'hw.perflevel{i}.{key}') for name, key in {
                'name': 'name', 'physical_cores': 'physicalcpu', 'logical_threads': 'logicalcpu',
                'l1_data_cache_bytes': 'l1dcachesize', 'l2_cache_bytes': 'l2cachesize',
                'cores_per_l2': 'cpusperl2'}.items()}
            if any(v is not None for v in level.values()):
                levels.append(level)
        p['performance_levels'] = levels
        try:
            hardware = json.loads(subprocess.check_output(
                ['/usr/sbin/system_profiler', 'SPHardwareDataType', '-json'], text=True))['SPHardwareDataType'][0]
            selected = {key: hardware.get(key) for key in ('chip_type', 'machine_model', 'number_processors', 'physical_memory')}
            p['system_profiler_selected'] = selected
            p['cpu_model'] = p['cpu_model'] or selected.get('chip_type')
            p['hardware_model'] = p.get('hardware_model') or selected.get('machine_model')
            match = re.search(r'proc (\d+)', selected.get('number_processors') or '')
            if match and p['physical_cores'] is None:
                p['physical_cores'] = int(match.group(1))
                p['physical_cores_source'] = 'system_profiler SPHardwareDataType number_processors'
            if p['logical_threads'] is None:
                p['logical_threads'] = os.cpu_count()
                p['logical_threads_source'] = 'os.cpu_count()'
        except (OSError, subprocess.CalledProcessError, ValueError, KeyError, IndexError) as exc:
            errors['system_profiler'] = str(exc)
        p['microarchitecture'] = {'source': 'sysctl hw.cpufamily identifier',
                                  'cpufamily': sysctl('hw.cpufamily')}
        p['frequency_note'] = 'Apple Silicon may not expose a base clock; null means unavailable, not zero.'
    elif platform.system() == 'Linux':
        text = open('/proc/cpuinfo').read()
        fields = dict(re.findall(r'^([^:\n]+)\s*:\s*(.*)$', text, flags=re.M))
        p['cpu_model'] = fields.get('model name', fields.get('Processor'))
        p['logical_threads'] = os.cpu_count()
        topology = set()
        from pathlib import Path
        for cpu in Path('/sys/devices/system/cpu').glob('cpu[0-9]*'):
            try:
                topology.add(((cpu/'topology/physical_package_id').read_text().strip(),
                              (cpu/'topology/core_id').read_text().strip()))
            except OSError:
                pass
        p['physical_cores'] = len(topology) or None
        caches = []
        for d in Path('/sys/devices/system/cpu/cpu0/cache').glob('index*'):
            try:
                size = (d/'size').read_text().strip()
                scale = 1024 if size.endswith('K') else 1024**2 if size.endswith('M') else 1
                item = {'level': int((d/'level').read_text()), 'type': (d/'type').read_text().strip(),
                        'size_bytes': int(size.rstrip('KM'))*scale,
                        'line_bytes': int((d/'coherency_line_size').read_text())}
                caches.append(item)
                if item['type'] in ('Data', 'Unified'):
                    name = {1: 'l1_data_cache_bytes', 2: 'l2_cache_bytes', 3: 'l3_cache_bytes'}[item['level']]
                    p[name] = item['size_bytes']
                    p['cache_line_bytes'] = item['line_bytes']
            except (OSError, ValueError, KeyError):
                pass
        p['cache_records'] = caches
        p['microarchitecture'] = {k: fields.get(k) for k in ('vendor_id', 'cpu family', 'model', 'stepping')}
        p['frequency_note'] = 'Current cpu MHz is not a base frequency; not substituted.'
    p['smt_detected'] = (p['logical_threads'] > p['physical_cores']) if p['physical_cores'] and p['logical_threads'] else None
    p['raw_sysctl'] = raw
    p['required_telemetry_complete'] = all(p[k] is not None for k in (
        'cpu_model', 'physical_cores', 'logical_threads', 'cache_line_bytes',
        'l1_data_cache_bytes', 'l2_cache_bytes'))
    p['cache_note'] = 'Missing L3 may mean absent or unexposed; null is never treated as a measured zero.'
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--student-id', required=True)
    parser.add_argument('--results-dir')
    args = parser.parse_args()
    out = output_dir(args.results_dir)
    write_json(out/'salted_params.json', get_student_params(args.student_id))
    p = profile()
    if len(args.student_id) != 8:
        p['student_id_note'] = 'The supplied ID has %d digits; full ID used unchanged despite PDF saying 8.' % len(args.student_id)
    write_json(out/'host_profile.json', p)
    print('M0 telemetry complete:', p['required_telemetry_complete'], flush=True)


if __name__ == '__main__':
    main()
