"""
bootstrap_peer.py
----------------
Bootstrap DHT peer — layers 0-16, AddConstant(+1.0).

Binds DHT on 0.0.0.0:43300 (reachable from LAN).
Binds P2P on 0.0.0.0:44211.
Advertises the address stored in DHT using --host-ip
(use the machine's LAN IP when the client is on another machine).

Run (same-machine test):
    conda activate soARM
    python p2p/bootstrap_peer.py

Run (cross-machine, LAN IP 192.168.x.y):
    python p2p/bootstrap_peer.py --host-ip 192.168.x.y
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

DHT_PORT = 43300
P2P_PORT = 44211
HTTP_PORT = 8765
DHT_TTL = 3600.0
HANDLER_NAME = "tensor_forward"
DHT_KEY = "bootstrap_peer"


def tensor_to_bytes(t: torch.Tensor) -> bytes:
    arr = t.detach().cpu().float().numpy()
    header = struct.pack(">" + "i" * (1 + arr.ndim), arr.ndim, *arr.shape)
    return header + arr.tobytes()


def bytes_to_tensor(data: bytes) -> torch.Tensor:
    ndim = struct.unpack(">i", data[:4])[0]
    shape = struct.unpack(">" + "i" * ndim, data[4 : 4 + ndim * 4])
    arr = np.frombuffer(data[4 + ndim * 4 :], dtype=np.float32).reshape(shape)
    return torch.from_numpy(arr.copy())


class AddConstant(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"  [bootstrap_peer +{self.value}]  shape={tuple(x.shape)}")
        return x + self.value


async def main(host_ip: str, dht_port: int, http_port: int, p2p_port: int) -> None:
    print("=" * 55)
    print(f"bootstrap_peer  DHT:{dht_port}  P2P:{p2p_port}  advertise={host_ip}")
    print("=" * 55)

    print(f"\n[1] Starting bootstrap DHT on 0.0.0.0:{dht_port} ...")
    dht = hivemind.DHT(
        host_maddrs=[f"/ip4/0.0.0.0/tcp/{dht_port}"],
        start=True,
    )
    await asyncio.sleep(1.0)
    all_maddrs = dht.get_visible_maddrs()
    dht_maddr = next(
        (str(m) for m in all_maddrs if host_ip in str(m)),
        str(all_maddrs[0]),
    )
    print(f"    DHT bootstrap maddr: {dht_maddr}")

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = dht_maddr.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    httpd = socketserver.TCPServer(("0.0.0.0", http_port), _Handler)
    httpd.allow_reuse_address = True
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    print(f"    Serving bootstrap maddr on http://0.0.0.0:{http_port}/")

    print(f"\n[2] Starting P2P node on 0.0.0.0:{p2p_port} ...")
    module = AddConstant(1.0)
    p2p = await P2P.create(host_maddrs=[f"/ip4/0.0.0.0/tcp/{p2p_port}"])

    async def handle(stream_info, reader, writer):
        data = await P2P.receive_raw_data(reader)
        result = module(bytes_to_tensor(data))
        await P2P.send_raw_data(tensor_to_bytes(result), writer)
        writer.close()

    await p2p.add_binary_stream_handler(HANDLER_NAME, handle)

    maddr = f"/ip4/{host_ip}/tcp/{p2p_port}/p2p/{p2p.peer_id}"
    peer_id = str(p2p.peer_id)
    info = json.dumps({"maddr": maddr, "peer_id": peer_id, "layers": "0-16"})
    dht.store(DHT_KEY, info, expiration_time=get_dht_time() + DHT_TTL)
    print(f"    Registered in DHT — maddr: {maddr}")

    print("\n[bootstrap_peer ready — Ctrl-C to stop]\n")
    try:
        while True:
            await asyncio.sleep(DHT_TTL / 2)
            dht.store(DHT_KEY, info, expiration_time=get_dht_time() + DHT_TTL)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        print("\nShutting down bootstrap_peer ...")
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
    parser.add_argument("--dht-port", type=int, default=DHT_PORT)
    parser.add_argument("--http-port", type=int, default=HTTP_PORT)
    parser.add_argument("--p2p-port", type=int, default=P2P_PORT)
    args = parser.parse_args()
    asyncio.run(main(args.host_ip, args.dht_port, args.http_port, args.p2p_port))
