import os
import queue
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

BOOTSTRAP_RE = re.compile(r"DHT bootstrap maddr:\s*(\S+)")

E2E_BOOTSTRAP_DHT = 43400
E2E_BOOTSTRAP_HTTP = 18765
E2E_BOOTSTRAP_P2P = 44231


@dataclass
class ProcHandle:
    name: str
    process: subprocess.Popen
    lines: list[str]
    line_queue: "queue.Queue[str]"
    thread: threading.Thread


def _reader_thread(prefix: str, stream, lines: list[str], q: "queue.Queue[str]") -> None:
    try:
        for raw in stream:
            line = raw.rstrip("\n")
            lines.append(line)
            print(f"[{prefix}] {line}", flush=True)
            q.put(line)
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _port_listening_regex(port: int) -> re.Pattern[str]:
    return re.compile(rf":{port}\s+\(LISTEN\)\s*$")


def find_listeners_on_ports(ports: list[int]) -> list[tuple[str, int, int]]:
    if not ports:
        return []
    cmd = ["lsof", "-nP"]
    for port in ports:
        cmd.append(f"-iTCP:{port}")
    cmd.append("-sTCP:LISTEN")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0 and not proc.stdout.strip():
        return []

    listeners: list[tuple[str, int, int]] = []
    for line in proc.stdout.splitlines():
        if not line or line.startswith("COMMAND"):
            continue
        cols = line.split()
        if len(cols) < 2:
            continue
        command = cols[0]
        try:
            pid = int(cols[1])
        except ValueError:
            continue
        for port in ports:
            if _port_listening_regex(port).search(line):
                listeners.append((command, pid, port))
    return listeners


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_pid(pid: int, grace_s: float = 2.0) -> None:
    try:
        os.kill(pid, 15)
    except ProcessLookupError:
        return
    except PermissionError:
        return

    deadline = time.time() + grace_s
    while time.time() < deadline:
        if not _pid_exists(pid):
            return
        time.sleep(0.05)

    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        return
    except PermissionError:
        return


def cleanup_stale_p2pd(ports: list[int]) -> None:
    listeners = find_listeners_on_ports(ports)
    stale_p2pd = sorted({pid for command, pid, _ in listeners if command == "p2pd"})
    for pid in stale_p2pd:
        print(f"[orchestrator] cleaning stale p2pd pid={pid}", flush=True)
        terminate_pid(pid)


def assert_ports_clear(ports: list[int]) -> None:
    listeners = find_listeners_on_ports(ports)
    if not listeners:
        return
    details = ", ".join(f"{cmd}:{pid}@{port}" for cmd, pid, port in listeners)
    raise RuntimeError(
        f"Ports not clear after cleanup: {details}. "
        "If these are privileged/orphaned daemons, run: sudo pkill -f p2pd"
    )


def start_process(name: str, cmd: list[str], cwd: str, env: dict[str, str]) -> ProcHandle:
    print(f"[orchestrator] starting {name}: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    q: "queue.Queue[str]" = queue.Queue()
    thread = threading.Thread(target=_reader_thread, args=(name, proc.stdout, lines, q), daemon=True)
    thread.start()
    return ProcHandle(name=name, process=proc, lines=lines, line_queue=q, thread=thread)


def wait_for_pattern(
    handle: ProcHandle,
    pattern: re.Pattern[str],
    timeout_s: float,
) -> re.Match[str]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if handle.process.poll() is not None:
            raise RuntimeError(f"{handle.name} exited early with code {handle.process.returncode}")
        remaining = max(0.01, deadline - time.time())
        try:
            line = handle.line_queue.get(timeout=min(0.5, remaining))
        except queue.Empty:
            continue
        match = pattern.search(line)
        if match:
            return match
    raise TimeoutError(f"Timed out waiting for pattern {pattern.pattern!r} in {handle.name}")


def wait_for_line_contains(handle: ProcHandle, fragment: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if handle.process.poll() is not None:
            raise RuntimeError(f"{handle.name} exited early with code {handle.process.returncode}")
        remaining = max(0.01, deadline - time.time())
        try:
            line = handle.line_queue.get(timeout=min(0.5, remaining))
        except queue.Empty:
            continue
        if fragment in line:
            return
    raise TimeoutError(f"Timed out waiting for {fragment!r} from {handle.name}")


def stop_process(handle: Optional[ProcHandle], timeout_s: float = 10.0) -> None:
    if handle is None or handle.process.poll() is not None:
        return
    try:
        os.killpg(handle.process.pid, 15)
    except ProcessLookupError:
        return
    except PermissionError:
        handle.process.terminate()
    try:
        handle.process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(handle.process.pid, 9)
        except ProcessLookupError:
            pass
        except PermissionError:
            handle.process.kill()
        handle.process.wait(timeout=timeout_s)


def require_exit_zero(handle: ProcHandle) -> None:
    code = handle.process.poll()
    if code is None:
        return
    if code != 0:
        raise RuntimeError(f"{handle.name} exited with code {code}")


def build_dht_key(base: str, run_id: str) -> str:
    return f"{base}_{run_id}" if run_id else base
