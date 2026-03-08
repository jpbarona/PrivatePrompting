import argparse
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


BOOTSTRAP_RE = re.compile(r"DHT bootstrap maddr:\s*(\S+)")


@dataclass
class ProcHandle:
    name: str
    process: subprocess.Popen
    lines: list[str]
    line_queue: "queue.Queue[str]"
    thread: threading.Thread


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-command high-fidelity network e2e inference.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--host-ip", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="Hey! How are you feeling today?")
    parser.add_argument("--num-new-tokens", type=int, default=16)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--num-middle-partitions", type=int, default=2)
    parser.add_argument("--max-nbytes", type=int, default=7340032)
    parser.add_argument("--max-seq", type=int, default=2048)
    parser.add_argument("--w1-layer-start", type=int, default=2)
    parser.add_argument("--w1-layer-end", type=int, default=12)
    parser.add_argument("--w2-layer-start", type=int, default=12)
    parser.add_argument("--w2-layer-end", type=int, default=22)
    parser.add_argument("--dht-port-w1", type=int, default=43311)
    parser.add_argument("--dht-port-w2", type=int, default=43312)
    parser.add_argument("--dht-port-parent", type=int, default=43313)
    parser.add_argument("--p2p-port-w1", type=int, default=44221)
    parser.add_argument("--p2p-port-w2", type=int, default=44212)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--startup-timeout-s", type=float, default=120.0)
    parser.add_argument("--parent-timeout-s", type=float, default=1800.0)
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"repo root does not exist: {repo_root}")

    if args.host_ip.startswith("127."):
        raise ValueError("HOST_IP must be a LAN IP (192.168.x.x etc.), not loopback 127.x.x.x")

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    run_id = args.run_id or f"e2e_{uuid.uuid4().hex[:8]}"
    dht_key_w1 = build_dht_key("inference_w1", run_id)
    dht_key_w2 = build_dht_key("inference_w2", run_id)
    e2e_ports = [
        43300,
        args.dht_port_w1,
        args.dht_port_w2,
        args.dht_port_parent,
        44211,
        args.p2p_port_w1,
        args.p2p_port_w2,
    ]
    print(f"[orchestrator] run_id={run_id}", flush=True)

    peer0: Optional[ProcHandle] = None
    w2: Optional[ProcHandle] = None
    w1: Optional[ProcHandle] = None
    parent: Optional[ProcHandle] = None

    try:
        cleanup_stale_p2pd(e2e_ports)
        assert_ports_clear(e2e_ports)

        peer0_cmd = [
            args.python,
            "singleMachinep2p/peer0.py",
            "--host-ip",
            args.host_ip,
        ]
        peer0 = start_process("peer0", peer0_cmd, cwd=str(repo_root), env=env)
        bootstrap_match = wait_for_pattern(peer0, BOOTSTRAP_RE, timeout_s=args.startup_timeout_s)
        bootstrap_maddr = bootstrap_match.group(1)
        print(f"[orchestrator] bootstrap_maddr={bootstrap_maddr}", flush=True)
        wait_for_line_contains(peer0, "[peer0 ready", timeout_s=args.startup_timeout_s)

        w2_cmd = [
            args.python,
            "inference/worker.py",
            "--role",
            "w2",
            "--model-name",
            args.model_name,
            "--host-ip",
            args.host_ip,
            "--dht-port",
            str(args.dht_port_w2),
            "--p2p-port",
            str(args.p2p_port_w2),
            "--bootstrap-maddr",
            bootstrap_maddr,
            "--dht-key",
            dht_key_w2,
            "--handler-name",
            "inference_frame",
            "--max-nbytes",
            str(args.max_nbytes),
            "--layer-start",
            str(args.w2_layer_start),
            "--layer-end",
            str(args.w2_layer_end),
        ]
        w2 = start_process("w2", w2_cmd, cwd=str(repo_root), env=env)
        wait_for_line_contains(w2, "[w2] DHT=", timeout_s=args.startup_timeout_s)

        w1_cmd = [
            args.python,
            "inference/worker.py",
            "--role",
            "w1",
            "--model-name",
            args.model_name,
            "--host-ip",
            args.host_ip,
            "--dht-port",
            str(args.dht_port_w1),
            "--p2p-port",
            str(args.p2p_port_w1),
            "--bootstrap-maddr",
            bootstrap_maddr,
            "--dht-key",
            dht_key_w1,
            "--next-dht-key",
            dht_key_w2,
            "--handler-name",
            "inference_frame",
            "--max-nbytes",
            str(args.max_nbytes),
            "--layer-start",
            str(args.w1_layer_start),
            "--layer-end",
            str(args.w1_layer_end),
        ]
        w1 = start_process("w1", w1_cmd, cwd=str(repo_root), env=env)
        wait_for_line_contains(w1, "[w1] DHT=", timeout_s=args.startup_timeout_s)

        parent_cmd = [
            args.python,
            "inference/parent_client.py",
            "--model-name",
            args.model_name,
            "--prompt",
            args.prompt,
            "--num-new-tokens",
            str(args.num_new_tokens),
            "--k",
            str(args.k),
            "--num-middle-partitions",
            str(args.num_middle_partitions),
            "--host-ip",
            args.host_ip,
            "--dht-port",
            str(args.dht_port_parent),
            "--bootstrap-maddr",
            bootstrap_maddr,
            "--w1-dht-key",
            dht_key_w1,
            "--w2-dht-key",
            dht_key_w2,
            "--max-seq",
            str(args.max_seq),
        ]
        parent = start_process("parent", parent_cmd, cwd=str(repo_root), env=env)
        parent.process.wait(timeout=args.parent_timeout_s)
        parent.thread.join(timeout=2.0)
        require_exit_zero(parent)

        if not any(line.strip() == "match: True" for line in parent.lines):
            raise RuntimeError("Parent finished but did not report 'match: True'")

        if w1 and w1.process.wait(timeout=30.0) != 0:
            raise RuntimeError("w1 did not exit cleanly after SHUTDOWN")
        if w2 and w2.process.wait(timeout=30.0) != 0:
            raise RuntimeError("w2 did not exit cleanly after SHUTDOWN")

        print("[orchestrator] PASS: all processes exited 0 and parent reported match: True", flush=True)
        return 0
    except KeyboardInterrupt:
        print("[orchestrator] interrupted", flush=True)
        return 130
    except Exception as exc:
        print(f"[orchestrator] FAIL: {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        stop_process(parent)
        stop_process(w1)
        stop_process(w2)
        stop_process(peer0)
        cleanup_stale_p2pd(e2e_ports)


if __name__ == "__main__":
    sys.exit(main())
