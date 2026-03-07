"""
peer0.py
--------
Bootstrap DHT peer — layers 0-16, AddConstant(+1.0).

Binds DHT on 0.0.0.0:43300 (reachable from LAN).
Binds P2P on 0.0.0.0:44211.
Advertises the address stored in DHT using --host-ip
(use the machine's LAN IP when the client is on another machine).

Run (same-machine test):
    conda activate soARM
    python peer0.py

Run (cross-machine, LAN IP 192.168.x.y):
    python peer0.py --host-ip 192.168.x.y
"""

import argparse
import asyncio
import http.server
import json
import socketserver
import struct
import threading

import numpy as np
import torch
import torch.nn as nn
import hivemind
from hivemind.p2p import P2P
from hivemind.utils import get_dht_time

# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------

DHT_PORT = 43300
P2P_PORT = 44211
HTTP_PORT = 8765  # serves the DHT bootstrap maddr to peer1 and the client
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
        print(f"  [peer0 +{self.value}]  shape={tuple(x.shape)}")
        return x + self.value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(host_ip: str) -> None:
    print("=" * 55)
    print(f"peer0  DHT:{DHT_PORT}  P2P:{P2P_PORT}  advertise={host_ip}")
    print("=" * 55)

    # ── DHT bootstrap ───────────────────────────────────────────────────────
    # Bind on 0.0.0.0 so the client on another machine can reach port 43300.
    print(f"\n[1] Starting bootstrap DHT on 0.0.0.0:{DHT_PORT} ...")
    dht = hivemind.DHT(
        host_maddrs=[f"/ip4/0.0.0.0/tcp/{DHT_PORT}"],
        start=True,
    )
    await asyncio.sleep(1.0)
    # Use the maddr that contains the advertised host_ip so remote peers work.
    all_maddrs = dht.get_visible_maddrs()
    dht_maddr = next(
        (str(m) for m in all_maddrs if host_ip in str(m)),
        str(all_maddrs[0]),
    )
    print(f"    DHT bootstrap maddr: {dht_maddr}")

    # ── Tiny HTTP server — serves dht_maddr to peer1 and the remote client ──
    # Uses only stdlib; no aiohttp / requests needed.
    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = dht_maddr.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):  # silence request logs
            pass

    httpd = socketserver.TCPServer(("0.0.0.0", HTTP_PORT), _Handler)
    httpd.allow_reuse_address = True
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    print(f"    Serving bootstrap maddr on http://0.0.0.0:{HTTP_PORT}/")

    # ── P2P node ────────────────────────────────────────────────────────────
    print(f"\n[2] Starting P2P node on 0.0.0.0:{P2P_PORT} ...")
    module = AddConstant(1.0)
    p2p = await P2P.create(host_maddrs=[f"/ip4/0.0.0.0/tcp/{P2P_PORT}"])

    async def handle(stream_info, reader, writer):
        data = await P2P.receive_raw_data(reader)
        result = module(bytes_to_tensor(data))
        await P2P.send_raw_data(tensor_to_bytes(result), writer)
        writer.close()

    await p2p.add_binary_stream_handler(HANDLER_NAME, handle)

    # Advertise the *host_ip* address so remote clients can reach us.
    maddr = f"/ip4/{host_ip}/tcp/{P2P_PORT}/p2p/{p2p.peer_id}"
    peer_id = str(p2p.peer_id)
    info = json.dumps({"maddr": maddr, "peer_id": peer_id, "layers": "0-16"})
    dht.store("peer0", info, expiration_time=get_dht_time() + DHT_TTL)
    print(f"    Registered in DHT — maddr: {maddr}")

    # ── Keep alive ──────────────────────────────────────────────────────────
    print("\n[peer0 ready — Ctrl-C to stop]\n")
    try:
        while True:
            # Refresh DHT entry before it expires
            await asyncio.sleep(DHT_TTL / 2)
            dht.store("peer0", info, expiration_time=get_dht_time() + DHT_TTL)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        print("\nShutting down peer0 ...")
        httpd.shutdown()
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
