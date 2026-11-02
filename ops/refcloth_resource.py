"""Host-wide admission and recovery for this study; never changes system limits."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

GIB = 1024 ** 3
BASE = Path('/dev/shm/refcloth_guard_%d' % os.getuid())
_pause = False


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.%d.tmp' % os.getpid())
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


@contextmanager
def locked():
    BASE.mkdir(exist_ok=True)
    with (BASE / 'ledger.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def identity(pid):
    try:
        root = Path('/proc/%d' % pid)
        stat = (root / 'stat').read_text().rsplit(') ', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return dict(pid=pid, ticks=int(stat[19]), uid=root.stat().st_uid,
                    cmd=(root / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace'))
    except (OSError, ValueError, IndexError):
        return None


def matches(entry):
    current = identity(entry['pid'])
    return bool(current and current['uid'] == os.getuid() == entry['uid']
                and current['ticks'] == entry['ticks']
                and entry['script'] in current['cmd']
                and '/code/tex3d/' in entry['script'])


def register(pid, script, kind, gpu, budget, label, root, phase='loading', started=None):
    current = identity(pid)
    assert current and current['uid'] == os.getuid() and str(script) in current['cmd']
    assert '/code/tex3d/' in str(script)
    token = '%d_%d' % (pid, current['ticks'])
    entry = dict(current, script=str(script), kind=kind, gpu=gpu, budget=budget,
                 label=label, root=str(root), phase=phase,
                 started=started or time.time(), token=token)
    with locked():
        entries = read(BASE / 'jobs.json', {})
        previous = entries.get(token)
        if previous and matches(previous):
            entry['phase'] = previous['phase']
            if 'ready_at' in previous:
                entry['ready_at'] = previous['ready_at']
        entries[token] = entry
        write(BASE / 'jobs.json', entries)
    return token


def active():
    return [v for v in read(BASE / 'jobs.json', {}).values() if matches(v)]


def mark_ready():
    """Called only after model and immutable inputs have finished loading."""
    current = identity(os.getpid())
    if not current:
        return
    token = '%d_%d' % (os.getpid(), current['ticks'])
    with locked():
        entries = read(BASE / 'jobs.json', {})
        if token in entries and matches(entries[token]):
            entries[token].update(phase='warm', ready_at=time.time())
            write(BASE / 'jobs.json', entries)


def install_pause_handler():
    def request(signum, frame):
        global _pause
        _pause = True
    signal.signal(signal.SIGTERM, request)


def pause_requested():
    return _pause


def resource_stopped(pid, ticks=None):
    events = read(BASE / 'stops.json', {})
    return any(e['pid'] == pid and (ticks is None or e['ticks'] == ticks) for e in events.values())


def _number(path):
    try:
        value = Path(path).read_text().strip()
        return None if value == 'max' else int(value)
    except (OSError, ValueError):
        return None


def memory_groups(proc='/proc/self/cgroup', v2root='/sys/fs/cgroup', v1root='/sys/fs/cgroup/memory'):
    """Inspect every memory ancestor, including the shared user.slice limit."""
    groups = []
    for row in Path(proc).read_text().splitlines():
        _, controllers, relative = row.split(':', 2)
        if controllers == '':
            version, mount = 2, Path(v2root)
        elif 'memory' in controllers.split(','):
            version, mount = 1, Path(v1root)
        else:
            continue
        node = mount / relative.lstrip('/')
        while True:
            if version == 2:
                used = _number(node / 'memory.current')
                high = _number(node / 'memory.high')
                hard = _number(node / 'memory.max')
            else:
                used = _number(node / 'memory.usage_in_bytes')
                high = _number(node / 'memory.soft_limit_in_bytes')
                hard = _number(node / 'memory.limit_in_bytes')
            limits = [v for v in [high, hard] if v is not None and v < 2 ** 60]
            if used is not None:
                cache_credit = 0
                if version == 2:
                    try:
                        stat = {r.split()[0]: int(r.split()[1]) for r in (node / 'memory.stat').read_text().splitlines()}
                        # Credit only half of clean, unmapped file cache. Dirty,
                        # writeback and mapped working pages receive no credit.
                        clean = max(0, stat.get('file', 0) - stat.get('file_mapped', 0)
                                    - stat.get('file_dirty', 0) - stat.get('file_writeback', 0))
                        cache_credit = clean // 2
                    except OSError:
                        pass
                headroom = min(limits) - used if limits else None
                groups.append(dict(path=str(node), version=version, used=used,
                                   high=high, hard=hard,
                                   headroom=headroom, cache_credit=cache_credit,
                                   effective_headroom=headroom + cache_credit if headroom is not None else None))
            if node == mount:
                break
            node = node.parent
    return groups


def _psi(name):
    try:
        return {r.split()[0]: {k: float(v) for k, v in
                (s.split('=') for s in r.split()[1:])}
                for r in Path('/proc/pressure' , name).read_text().splitlines()}
    except OSError:
        return {}


def _counters():
    cpu = [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
    vm = {r.split()[0]: int(r.split()[1]) for r in Path('/proc/vmstat').read_text().splitlines()}
    io = 0
    physical = {p.name for p in Path('/sys/block').iterdir()
                if not p.name.startswith(('loop', 'ram', 'dm-', 'md'))}
    for row in Path('/proc/diskstats').read_text().splitlines():
        p = row.split()
        if p[2] in physical:
            io += (int(p[5]) + int(p[9])) * 512
    return dict(cpu_total=sum(cpu[:8]), cpu_wait=cpu[4], disk_bytes=io,
                swap_pages=vm.get('pswpin', 0) + vm.get('pswpout', 0))


def snapshot(previous=None):
    now = time.time()
    mem = {r.split(':')[0]: int(r.split()[1]) * 1024
           for r in Path('/proc/meminfo').read_text().splitlines()}
    groups = memory_groups()
    headrooms = [g['headroom'] for g in groups if g['headroom'] is not None]
    effective = [g['effective_headroom'] for g in groups if g['effective_headroom'] is not None]
    counters = _counters()
    out = dict(time=now, available=mem['MemAvailable'], groups=groups,
               headroom=min(headrooms) if headrooms else None,
               effective_headroom=min(effective) if effective else None,
               memory_psi=_psi('memory'), io_psi=_psi('io'), counters=counters,
               iowait_percent=None, io_mib_s=None, swap_mib_s=None)
    if previous and now > previous['time'] and previous.get('counters'):
        old = previous['counters']; seconds = now - previous['time']
        out['iowait_percent'] = 100 * max(0, counters['cpu_wait'] - old['cpu_wait']) / max(1, counters['cpu_total'] - old['cpu_total'])
        out['io_mib_s'] = max(0, counters['disk_bytes'] - old['disk_bytes']) / seconds / 1024 ** 2
        out['swap_mib_s'] = max(0, counters['swap_pages'] - old['swap_pages']) * os.sysconf('SC_PAGE_SIZE') / seconds / 1024 ** 2
    return out


def _pressure(s, field, mode):
    return s.get(field, {}).get(mode, {}).get('avg10', 0.)


def admission_reasons(s, cold_gib=24):
    reasons = []
    if s['available'] < 50 * GIB:
        reasons.append('host_memory_below_50GiB')
    headroom = s.get('effective_headroom')
    if headroom is None:
        headroom = s['headroom']
    if headroom is not None and headroom < cold_gib * GIB:
        reasons.append('ancestor_headroom_below_%dGiB' % cold_gib)
    for field, mode, limit in [('memory_psi', 'full', 1.), ('memory_psi', 'some', 2.)]:
        if _pressure(s, field, mode) >= limit:
            reasons.append('%s_%s_high' % (field, mode))
    if any(s[k] is None for k in ['iowait_percent', 'io_mib_s', 'swap_mib_s']):
        reasons.append('waiting_for_resource_counter_interval')
    if io_stalled(s) and (s.get('iowait_percent') or 0) >= 50.:
        reasons.append('extreme_IO_wait')
    if (s.get('io_mib_s') or 0) >= 1024. and (s.get('iowait_percent') or 0) >= 50.:
        reasons.append('high_IO_throughput_with_CPU_wait')
    if (s.get('swap_mib_s') or 0) >= 1.:
        reasons.append('active_swap')
    return reasons


def io_stalled(s):
    # Global IO PSI alone can be high during normal cold model loading while
    # this worker is making progress. Require substantial CPU IO wait too.
    return (_pressure(s, 'io_psi', 'full') >= 30.
            and (s.get('iowait_percent') or 0) >= 20.)


def memory_critical(s):
    return s['available'] < 10 * GIB or _pressure(s, 'memory_psi', 'full') >= 10.


def severe(s):
    return memory_critical(s) or io_stalled(s)


def progress_age(entry, now):
    root = Path(entry['root']); name = entry['label'].split(':', 1)[1]
    path = root / ('logs/' + name + '.log' if entry['kind'] == 'source'
                   else 'evaluation_' + name + '.log')
    latest = max(entry['started'], entry.get('ready_at', 0))
    try:
        latest = max(latest, path.stat().st_mtime)
    except OSError:
        pass
    return max(0., now - latest)


def should_pause(entry, s):
    if memory_critical(s):
        return True
    grace = 600 if entry['phase'] == 'loading' else 300
    return io_stalled(s) and progress_age(entry, s['time']) >= grace


def can_launch(cold_gib=24, max_models=None):
    status = read(BASE / 'health.json', {})
    if time.time() - status.get('time', 0) > 25:
        return False, ['health_monitor_missing_or_stale']
    reasons = admission_reasons(status, cold_gib=cold_gib)
    if not status.get('healthy_since') or time.time() - status['healthy_since'] < 60:
        reasons.append('waiting_for_60s_stable_health')
    entries = active()
    if any(e['phase'] == 'loading' for e in entries):
        reasons.append('another_model_loading')
    limit = max_models if max_models is not None else (2 if 'zgc-33-9' in os.uname().nodename else 5)
    if len(entries) >= limit:
        reasons.append('managed_model_count_limit')
    return not reasons, reasons


def gpu_headroom(gpu):
    free = {int(r.split(',')[0]): int(r.split(',')[1]) for r in subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], universal_newlines=True).splitlines()}
    used = {int(r.split(',')[0]): int(r.split(',')[1]) for r in subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], universal_newlines=True).splitlines()
        if r.split(',')[1].strip().isdigit()}
    return free[gpu] - sum(max(0, e['budget'] - used.get(e['pid'], 0)) for e in active() if e['gpu'] == gpu)


def stop_entry(entry, sig):
    """Use a pidfd when available; identity, owner and command must all match."""
    fd = None
    try:
        if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
            fd = os.pidfd_open(entry['pid'])
        if not matches(entry):
            return False
        if fd is None:
            os.kill(entry['pid'], sig)
        else:
            signal.pidfd_send_signal(fd, sig)
        return True
    except ProcessLookupError:
        return False
    finally:
        if fd is not None:
            os.close(fd)


def monitor():
    BASE.mkdir(exist_ok=True)
    singleton = (BASE / 'monitor.lock').open('a')
    try:
        fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    previous = None; healthy_since = None; severe_since = None; last_stop = 0.
    while True:
        s = snapshot(previous); previous = s
        # Source loaders peak below 6GiB RSS in the measured frozen recipe;
        # reserve 10GiB for them. Evaluation keeps its separate 24GiB check.
        reasons = admission_reasons(s, cold_gib=10)
        healthy_since = (healthy_since or s['time']) if not reasons else None
        severe_since = (severe_since or s['time']) if severe(s) else None
        entries = active()
        with locked():
            stops = read(BASE / 'stops.json', {})
            for entry in entries:
                stopped = stops.get(entry['token'])
                if stopped and s['time'] - stopped['time'] >= 45 and not stopped.get('kill_sent'):
                    stopped['kill_sent'] = stop_entry(entry, signal.SIGKILL)
            if severe_since and s['time'] - severe_since >= 30 and s['time'] - last_stop >= 30:
                candidates = [e for e in entries if e['token'] not in stops and should_pause(e, s)]
                if candidates:
                    entry = max(candidates, key=lambda e: (e['phase'] == 'loading', e['started']))
                    record = dict(entry, time=s['time'], reason='sustained_resource_pressure', health=s)
                    # Write the recovery record before delivering SIGTERM.
                    stops[entry['token']] = record
                    write(BASE / 'stops.json', stops)
                    record['term_sent'] = stop_entry(entry, signal.SIGTERM)
                    print('RESOURCE_STOP', entry['label'], entry['pid'], flush=True)
                    last_stop = s['time']; healthy_since = None
            write(BASE / 'stops.json', stops)
        write(BASE / 'health.json', dict(s, healthy_since=healthy_since,
              severe_since=severe_since, admission_reasons=reasons,
              managed=[dict(pid=e['pid'], label=e['label'], phase=e['phase']) for e in entries]))
        time.sleep(5)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['monitor', 'snapshot'])
    args = p.parse_args()
    if args.command == 'monitor':
        monitor()
    else:
        print(json.dumps(snapshot(), indent=2))
