"""
client.py
---------
Connects to the peer network, sends a zero tensor through both peers,
and asserts that the final mean is 11.0.

Usage:
    conda activate soARM
    python client.py <host_ip>

    <host_ip>  — LAN IP of the machine running peer0.py / peer1.py
                 e.g.  python client.py 192.168.1.42

For a same-machine test:
    python client.py 127.0.0.1
"""

import asyncio
import json
import struct
import sys
import time
import urllib.request

import numpy as np
import torch
from hivemind.p2p import P2P
from hivemind.p2p.p2p_daemon_bindings.datastructures import PeerID
import hivemind

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HTTP_PORT = 8765  # peer0's bootstrap-maddr HTTP endpoint
HIDDEN_SIZE = 64
SEQ_LEN = 8
HANDLER_NAME = "tensor_forward"

# ---------------------------------------------------------------------------
# Tensor serialisation  (identical to test_peers.py / peer0.py / peer1.py)
# ---------------------------------------------------------------------------


def tensor_to_bytes(t: torch.Tensor) -> bytes:
    arr = t.detach().cpu().float().numpy()
    header = struct.pack(">" + "i" * (1 + arr.ndim), arr.ndim, *arr.shape)
    return header + arr.tobytes()


def bytes_to_tensor(data: bytes) -> torch.Tensor:
    ndim = struct.unpack(">i", data[:4])[0]
    shape = struct.unpack(">" + "i" * ndim, data[4 : 4 + ndim * 4])
    arr = np.frombuffer(data[4 + ndim * 4 :], dtype=np.float32).reshape(shape)
    return torch.from_numpy(arr.copy())


# ---------------------------------------------------------------------------
# DHT helper
# ---------------------------------------------------------------------------


def get_peer(dht: hivemind.DHT, key: str, retries: int = 15) -> dict:
    """Poll the DHT until *key* is found, then return its JSON payload."""
    for attempt in range(1, retries + 1):
        result = dht.get(key, latest=True)
        if result is not None:
            return json.loads(result.value)
        print(f"    [{attempt}/{retries}] '{key}' not found yet — retrying ...")
        time.sleep(0.5)
    raise RuntimeError(
        f"Peer '{key}' not found in DHT after {retries} attempts. "
        "Is the peer running and reachable?"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(host_ip: str) -> None:
    print("=" * 55)
    print(f"client  →  {host_ip}")
    print("=" * 55)

    # ── Fetch bootstrap maddr from peer0's HTTP endpoint ────────────────────
    # peer0 serves its full /ip4/.../tcp/.../p2p/<id> maddr on HTTP_PORT.
    # A bare /ip4/.../tcp/... is rejected by hivemind as an invalid p2p maddr.
    url = f"http://{host_ip}:{HTTP_PORT}/"
    print(f"\n[1] Fetching bootstrap maddr from {url} ...")
    boot_maddr = None
    for attempt in range(1, 16):
        try:
            boot_maddr = urllib.request.urlopen(url, timeout=3).read().decode().strip()
            break
        except Exception as exc:
            print(f"    [{attempt}/15] {exc} — retrying ...")
            time.sleep(1.0)
    if boot_maddr is None:
        raise RuntimeError(
            f"Could not reach {url} — is peer0.py running and reachable?"
        )
    print(f"    Bootstrap maddr: {boot_maddr}")

    print("    Joining DHT ...")
    dht = hivemind.DHT(initial_peers=[boot_maddr], start=True)
    await asyncio.sleep(1.0)
    print("    DHT joined.")

    # ── Discover peers ───────────────────────────────────────────────────────
    print("\n[2] Discovering peers ...")
    info0 = get_peer(dht, "peer0")
    info1 = get_peer(dht, "peer1")
    print(f"    peer0 — layers={info0['layers']}  maddr={info0['maddr']}")
    print(f"    peer1 — layers={info1['layers']}  maddr={info1['maddr']}")

    # ── Create client P2P node ───────────────────────────────────────────────
    print("\n[3] Creating client P2P node ...")
    client_p2p = await P2P.create(
        initial_peers=[info0["maddr"], info1["maddr"]],
        host_maddrs=["/ip4/0.0.0.0/tcp/0"],
    )
    peer_id0 = PeerID.from_base58(info0["peer_id"])
    peer_id1 = PeerID.from_base58(info1["peer_id"])
    print("    P2P node ready.")

    # ── Build input tensor ───────────────────────────────────────────────────
    x = torch.zeros(SEQ_LEN, HIDDEN_SIZE)
    print(f"\n[4] Input tensor: shape={tuple(x.shape)}  mean={x.mean():.4f}")

    # ── Forward through peer 0 ───────────────────────────────────────────────
    print("\n[5] Sending to peer0 (layers 0-16, expects +1.0) ...")
    _, r0, w0 = await client_p2p.call_binary_stream_handler(peer_id0, HANDLER_NAME)
    await P2P.send_raw_data(tensor_to_bytes(x), w0)
    y = bytes_to_tensor(await P2P.receive_raw_data(r0))
    w0.close()
    print(f"    Result: mean={y.mean():.4f}  (expected 1.0)")
    assert abs(y.mean().item() - 1.0) < 1e-4, f"peer0 wrong: got {y.mean():.4f}"
    print("    ✓ peer0 assertion passed")

    # ── Forward through peer 1 ───────────────────────────────────────────────
    print("\n[6] Sending to peer1 (layers 17-32, expects +10.0) ...")
    _, r1, w1 = await client_p2p.call_binary_stream_handler(peer_id1, HANDLER_NAME)
    await P2P.send_raw_data(tensor_to_bytes(y), w1)
    z = bytes_to_tensor(await P2P.receive_raw_data(r1))
    w1.close()
    print(f"    Result: mean={z.mean():.4f}  (expected 11.0)")
    assert abs(z.mean().item() - 11.0) < 1e-4, f"peer1 wrong: got {z.mean():.4f}"
    print("    ✓ peer1 assertion passed")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("ALL CHECKS PASSED")
    print(f"  zeros → peer0(+1.0) → peer1(+10.0) → mean={z.mean():.2f}")
    print("=" * 55)

    # ── Cleanup ──────────────────────────────────────────────────────────────
    await client_p2p.shutdown()
    dht.shutdown()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <host_ip>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
