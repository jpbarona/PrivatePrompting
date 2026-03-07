"""
peer1.py
--------
Secondary DHT peer — layers 17-32, AddConstant(+10.0).

Joins DHT via 127.0.0.1:43300 (peer0's bootstrap, same machine).
Binds P2P on 0.0.0.0:44212.
Advertises the address stored in DHT using --host-ip.

Run (same-machine test):
    conda activate soARM
    python peer1.py

Run (cross-machine, LAN IP 192.168.x.y):
    python peer1.py --host-ip 192.168.x.y
"""

import argparse
import asyncio
import json
import struct
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import hivemind
from hivemind.p2p import P2P
from hivemind.utils import get_dht_time

# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------

BOOT_HTTP_URL = "http://127.0.0.1:8765/"  # peer0's maddr HTTP endpoint
DHT_PORT = 43301
P2P_PORT = 44212
DHT_TTL = 3600.0
HANDLER_NAME = "tensor_forward"

# ---------------------------------------------------------------------------
# Tensor serialisation  (identical to test_peers.py)
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
# Module
# ---------------------------------------------------------------------------


class AddConstant(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"  [peer1 +{self.value}]  shape={tuple(x.shape)}")
        return x + self.value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(host_ip: str) -> None:
    print("=" * 55)
    print(f"peer1  DHT:{DHT_PORT}  P2P:{P2P_PORT}  advertise={host_ip}")
    print("=" * 55)

    # ── Fetch peer0's full DHT maddr via its HTTP endpoint ──────────────────
    # A bare /ip4/.../tcp/... is rejected by hivemind; we need the full
    # /ip4/.../tcp/.../p2p/<peer_id> form, which peer0 serves over HTTP.
    print(f"\n[1] Fetching bootstrap maddr from {BOOT_HTTP_URL} ...")
    boot_maddr = None
    for attempt in range(1, 31):
        try:
            boot_maddr = (
                urllib.request.urlopen(BOOT_HTTP_URL, timeout=3).read().decode().strip()
            )
            break
        except Exception as exc:
            print(f"    [{attempt}/30] {exc} — retrying ...")
            time.sleep(0.5)
    if boot_maddr is None:
        raise RuntimeError(f"Could not reach {BOOT_HTTP_URL} — is peer0.py running?")
    print(f"    Joining DHT via {boot_maddr} ...")
    dht = hivemind.DHT(
        host_maddrs=[f"/ip4/0.0.0.0/tcp/{DHT_PORT}"],
        initial_peers=[boot_maddr],
        start=True,
    )
    await asyncio.sleep(0.5)
    print(f"    DHT visible maddr: {dht.get_visible_maddrs()[0]}")

    # ── P2P node ────────────────────────────────────────────────────────────
    print(f"\n[2] Starting P2P node on 0.0.0.0:{P2P_PORT} ...")
    module = AddConstant(10.0)
    p2p = await P2P.create(host_maddrs=[f"/ip4/0.0.0.0/tcp/{P2P_PORT}"])

    async def handle(stream_info, reader, writer):
        data = await P2P.receive_raw_data(reader)
        result = module(bytes_to_tensor(data))
        await P2P.send_raw_data(tensor_to_bytes(result), writer)
        writer.close()

    await p2p.add_binary_stream_handler(HANDLER_NAME, handle)

    maddr = f"/ip4/{host_ip}/tcp/{P2P_PORT}/p2p/{p2p.peer_id}"
    peer_id = str(p2p.peer_id)
    info = json.dumps({"maddr": maddr, "peer_id": peer_id, "layers": "17-32"})
    dht.store("peer1", info, expiration_time=get_dht_time() + DHT_TTL)
    print(f"    Registered in DHT — maddr: {maddr}")

    # ── Keep alive ──────────────────────────────────────────────────────────
    print("\n[peer1 ready — Ctrl-C to stop]\n")
    try:
        while True:
            await asyncio.sleep(DHT_TTL / 2)
            dht.store("peer1", info, expiration_time=get_dht_time() + DHT_TTL)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        print("\nShutting down peer1 ...")
        await p2p.shutdown()
        dht.shutdown()
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host-ip",
        default="127.0.0.1",
        help="IP to advertise in DHT (use LAN IP for cross-machine access)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.host_ip))
