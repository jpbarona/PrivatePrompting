import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.e2e.orchestrate import (
    BOOTSTRAP_RE,
    E2E_BOOTSTRAP_DHT,
    E2E_BOOTSTRAP_HTTP,
    E2E_BOOTSTRAP_P2P,
    ProcHandle,
    assert_ports_clear,
    build_dht_key,
    cleanup_stale_p2pd,
    start_process,
    stop_process,
    wait_for_line_contains,
    wait_for_pattern,
)


@dataclass
class E2EServices:
    repo_root: Path
    python: str
    bootstrap_maddr: str
    dht_key_w1: str
    dht_key_w2: str
    bootstrap_handle: ProcHandle
    w1_handle: ProcHandle
    w2_handle: ProcHandle
    e2e_ports: list[int]
    model_name: str
    prompt: str
    num_new_tokens: int
    k: int
    num_middle_partitions: int
    host_ip: str
    dht_port_parent: int
    max_seq: int
    parent_timeout_s: float


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--host-ip", required=True, dest="host_ip", help="LAN IP (not 127.x)")
    parser.addoption("--python", default=None, dest="python")
    parser.addoption("--repo-root", default=None, dest="repo_root")
    parser.addoption("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct", dest="model_name")
    parser.addoption("--prompt", default="Hey! How are you feeling today?", dest="prompt")
    parser.addoption("--num-new-tokens", type=int, default=16, dest="num_new_tokens")
    parser.addoption("--k", type=int, default=2, dest="k")
    parser.addoption("--num-middle-partitions", type=int, default=2, dest="num_middle_partitions")
    parser.addoption("--max-nbytes", type=int, default=7340032, dest="max_nbytes")
    parser.addoption("--max-seq", type=int, default=2048, dest="max_seq")
    parser.addoption("--w1-layer-start", type=int, default=2, dest="w1_layer_start")
    parser.addoption("--w1-layer-end", type=int, default=12, dest="w1_layer_end")
    parser.addoption("--w2-layer-start", type=int, default=12, dest="w2_layer_start")
    parser.addoption("--w2-layer-end", type=int, default=22, dest="w2_layer_end")
    parser.addoption("--dht-port-w1", type=int, default=43311, dest="dht_port_w1")
    parser.addoption("--dht-port-w2", type=int, default=43312, dest="dht_port_w2")
    parser.addoption("--dht-port-parent", type=int, default=43313, dest="dht_port_parent")
    parser.addoption("--p2p-port-w1", type=int, default=44221, dest="p2p_port_w1")
    parser.addoption("--p2p-port-w2", type=int, default=44212, dest="p2p_port_w2")
    parser.addoption("--run-id", default="", dest="run_id")
    parser.addoption("--startup-timeout-s", type=float, default=120.0, dest="startup_timeout_s")
    parser.addoption("--parent-timeout-s", type=float, default=1800.0, dest="parent_timeout_s")


@pytest.fixture(scope="session")
def e2e_services(request: pytest.FixtureRequest) -> E2EServices:
    import sys

    opts = request.config
    host_ip = opts.getoption("host_ip")
    if host_ip.startswith("127."):
        raise ValueError("host-ip must be a LAN IP (192.168.x.x etc.), not loopback 127.x.x.x")

    repo_root = Path(opts.getoption("repo_root") or str(Path(__file__).resolve().parents[2])).resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"repo root does not exist: {repo_root}")

    python = opts.getoption("python") or sys.executable
    run_id = opts.getoption("run_id") or f"e2e_{uuid.uuid4().hex[:8]}"
    dht_key_w1 = build_dht_key("inference_w1", run_id)
    dht_key_w2 = build_dht_key("inference_w2", run_id)
    dht_port_w1 = opts.getoption("dht_port_w1")
    dht_port_w2 = opts.getoption("dht_port_w2")
    dht_port_parent = opts.getoption("dht_port_parent")
    p2p_port_w1 = opts.getoption("p2p_port_w1")
    p2p_port_w2 = opts.getoption("p2p_port_w2")
    e2e_ports = [
        E2E_BOOTSTRAP_DHT,
        E2E_BOOTSTRAP_HTTP,
        E2E_BOOTSTRAP_P2P,
        dht_port_w1,
        dht_port_w2,
        dht_port_parent,
        p2p_port_w1,
        p2p_port_w2,
    ]
    startup_timeout_s = opts.getoption("startup_timeout_s")
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    cleanup_stale_p2pd(e2e_ports)
    assert_ports_clear(e2e_ports)

    bootstrap_peer_cmd = [
        python,
        "p2p/bootstrap_peer.py",
        "--host-ip",
        host_ip,
        "--dht-port",
        str(E2E_BOOTSTRAP_DHT),
        "--http-port",
        str(E2E_BOOTSTRAP_HTTP),
        "--p2p-port",
        str(E2E_BOOTSTRAP_P2P),
    ]
    bootstrap_peer = start_process("bootstrap_peer", bootstrap_peer_cmd, cwd=str(repo_root), env=env)
    bootstrap_match = wait_for_pattern(bootstrap_peer, BOOTSTRAP_RE, timeout_s=startup_timeout_s)
    bootstrap_maddr = bootstrap_match.group(1)
    wait_for_line_contains(bootstrap_peer, "[bootstrap_peer ready", timeout_s=startup_timeout_s)

    model_name = opts.getoption("model_name")
    max_nbytes = opts.getoption("max_nbytes")
    w1_layer_start = opts.getoption("w1_layer_start")
    w1_layer_end = opts.getoption("w1_layer_end")
    w2_layer_start = opts.getoption("w2_layer_start")
    w2_layer_end = opts.getoption("w2_layer_end")

    w2_cmd = [
        python,
        "inference/worker.py",
        "--role",
        "w2",
        "--model-name",
        model_name,
        "--host-ip",
        host_ip,
        "--dht-port",
        str(dht_port_w2),
        "--p2p-port",
        str(p2p_port_w2),
        "--bootstrap-maddr",
        bootstrap_maddr,
        "--dht-key",
        dht_key_w2,
        "--handler-name",
        "inference_frame",
        "--max-nbytes",
        str(max_nbytes),
        "--layer-start",
        str(w2_layer_start),
        "--layer-end",
        str(w2_layer_end),
    ]
    w2 = start_process("w2", w2_cmd, cwd=str(repo_root), env=env)
    wait_for_line_contains(w2, "[w2] DHT=", timeout_s=startup_timeout_s)

    w1_cmd = [
        python,
        "inference/worker.py",
        "--role",
        "w1",
        "--model-name",
        model_name,
        "--host-ip",
        host_ip,
        "--dht-port",
        str(dht_port_w1),
        "--p2p-port",
        str(p2p_port_w1),
        "--bootstrap-maddr",
        bootstrap_maddr,
        "--dht-key",
        dht_key_w1,
        "--next-dht-key",
        dht_key_w2,
        "--handler-name",
        "inference_frame",
        "--max-nbytes",
        str(max_nbytes),
        "--layer-start",
        str(w1_layer_start),
        "--layer-end",
        str(w1_layer_end),
    ]
    w1 = start_process("w1", w1_cmd, cwd=str(repo_root), env=env)
    wait_for_line_contains(w1, "[w1] DHT=", timeout_s=startup_timeout_s)

    try:
        yield E2EServices(
            repo_root=repo_root,
            python=python,
            bootstrap_maddr=bootstrap_maddr,
            dht_key_w1=dht_key_w1,
            dht_key_w2=dht_key_w2,
            bootstrap_handle=bootstrap_peer,
            w1_handle=w1,
            w2_handle=w2,
            e2e_ports=e2e_ports,
            model_name=model_name,
            prompt=opts.getoption("prompt"),
            num_new_tokens=opts.getoption("num_new_tokens"),
            k=opts.getoption("k"),
            num_middle_partitions=opts.getoption("num_middle_partitions"),
            host_ip=host_ip,
            dht_port_parent=dht_port_parent,
            max_seq=opts.getoption("max_seq"),
            parent_timeout_s=opts.getoption("parent_timeout_s"),
        )
    finally:
        stop_process(w1)
        stop_process(w2)
        stop_process(bootstrap_peer)
        cleanup_stale_p2pd(e2e_ports)
