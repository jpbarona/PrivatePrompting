import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

from protocol import KIND_ERROR, KIND_SHUTDOWN, decode_frame, encode_frame
from p2p_transport import call_handler, create_p2p_node, discover_worker, join_dht


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--bootstrap-maddr", required=True)
    p.add_argument("--host-ip", default="127.0.0.1")
    p.add_argument("--max-seq", type=int, default=128)
    p.add_argument("--num-new-tokens", type=int, default=4)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--max-nbytes", type=int, default=7340032)
    p.add_argument("--dht-port-w1", type=int, default=43321)
    p.add_argument("--dht-port-w2", type=int, default=43322)
    p.add_argument("--dht-port-parent", type=int, default=43323)
    p.add_argument("--p2p-port-w1", type=int, default=44221)
    p.add_argument("--p2p-port-w2", type=int, default=44222)
    p.add_argument("--w1-dht-key", default="inference_w1")
    p.add_argument("--w2-dht-key", default="inference_w2")
    return p.parse_args()


def wait_for_startup(seconds: float = 20.0):
    time.sleep(seconds)


def start_worker(args, role: str) -> subprocess.Popen:
    here = Path(__file__).parent
    dht_port = args.dht_port_w1 if role == "w1" else args.dht_port_w2
    p2p_port = args.p2p_port_w1 if role == "w1" else args.p2p_port_w2
    layer_start, layer_end = ((2, 12) if role == "w1" else (12, 22))
    cmd = [
        sys.executable,
        str(here / "worker.py"),
        "--role",
        role,
        "--model-name",
        args.model_name,
        "--host-ip",
        args.host_ip,
        "--dht-port",
        str(dht_port),
        "--p2p-port",
        str(p2p_port),
        "--bootstrap-maddr",
        args.bootstrap_maddr,
        "--dht-key",
        args.w1_dht_key if role == "w1" else args.w2_dht_key,
        "--max-nbytes",
        str(args.max_nbytes),
        "--layer-start",
        str(layer_start),
        "--layer-end",
        str(layer_end),
    ]
    if role == "w1":
        cmd.extend(["--next-dht-key", args.w2_dht_key])
    return subprocess.Popen(
        cmd,
        cwd=here,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def run_parent(args) -> str:
    here = Path(__file__).parent
    cmd = [
        sys.executable,
        str(here / "parent_client.py"),
        "--model-name",
        args.model_name,
        "--prompt",
        "Hey! How are you feeling today?",
        "--num-new-tokens",
        str(args.num_new_tokens),
        "--k",
        str(args.k),
        "--num-middle-partitions",
        "2",
        "--host-ip",
        args.host_ip,
        "--dht-port",
        str(args.dht_port_parent),
        "--bootstrap-maddr",
        args.bootstrap_maddr,
        "--w1-dht-key",
        args.w1_dht_key,
        "--w2-dht-key",
        args.w2_dht_key,
        "--max-seq",
        str(args.max_seq),
    ]
    output = subprocess.check_output(
        cmd,
        cwd=here,
        env=os.environ.copy(),
        text=True,
        stderr=subprocess.STDOUT,
    )
    if "match: True" not in output:
        raise AssertionError(f"Expected parity success. Output:\n{output}")
    return output


async def run_failure_probe(args):
    dht, _ = join_dht(
        dht_port=args.dht_port_parent + 10,
        bootstrap_maddr=args.bootstrap_maddr,
        host_ip=args.host_ip,
    )
    p2p = await create_p2p_node(p2p_port=0)
    w1 = discover_worker(dht, key=args.w1_dht_key)
    try:
        bad_frame = encode_frame({"kind": "unknown_kind"})
        reply = await call_handler(
            p2p=p2p,
            peer_id=w1.peer_id,
            peer_maddr=w1.maddr,
            handler_name=w1.handler_name,
            payload_bytes=bad_frame,
        )
        frame, _ = decode_frame(reply, args.max_nbytes)
        if frame["kind"] != KIND_ERROR:
            raise AssertionError(f"Expected KIND_ERROR for invalid frame, got {frame}")

        shutdown_reply = await call_handler(
            p2p=p2p,
            peer_id=w1.peer_id,
            peer_maddr=w1.maddr,
            handler_name=w1.handler_name,
            payload_bytes=encode_frame({"kind": KIND_SHUTDOWN}),
        )
        shutdown_frame, _ = decode_frame(shutdown_reply, args.max_nbytes)
        if shutdown_frame["kind"] != KIND_SHUTDOWN:
            raise AssertionError(f"Expected shutdown ack, got {shutdown_frame}")
    finally:
        await p2p.shutdown()
        dht.shutdown()


def main():
    args = parse_args()
    w2 = start_worker(args, "w2")
    wait_for_startup()
    w1 = start_worker(args, "w1")
    wait_for_startup()
    try:
        out = run_parent(args)
        print("[happy-path] parent output validated")
        print(out)
    finally:
        if w1.poll() is None:
            w1.terminate()
        if w2.poll() is None:
            w2.terminate()
        w1.wait(timeout=30)
        w2.wait(timeout=30)

    w2 = start_worker(args, "w2")
    wait_for_startup()
    w1 = start_worker(args, "w1")
    wait_for_startup()
    try:
        asyncio.run(run_failure_probe(args))
        print("[failure-path] invalid frame and shutdown behavior validated")
    finally:
        if w1.poll() is None:
            w1.terminate()
        if w2.poll() is None:
            w2.terminate()
        w1.wait(timeout=30)
        w2.wait(timeout=30)


if __name__ == "__main__":
    main()
