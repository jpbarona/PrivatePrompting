"""
test_peers.py
--------------
Tests hivemind P2P tensor passing using only hivemind.DHT and the raw P2P
layer.  No MoE components (ExpertBackend / ModuleBackend / Server /
RemoteExpert) are used.

What is tested
--------------
  1. Two DHT nodes form a network on localhost.
  2. Peer 0 announces "layers 0-16" on the DHT.
  3. Peer 1 announces "layers 17-32" on the DHT.
  4. A client joins the DHT, discovers both peers and sends a tensor to peer 0.
  5. Peer 0 runs AddConstant(+1.0) and returns the result.
  6. Client forwards the result to peer 1.
  7. Peer 1 runs AddConstant(+10.0) and returns the result.
  8. Client asserts final tensor mean == 11.0.

Transport
---------
  - hivemind.DHT          peer discovery  (store / get)
  - hivemind.P2P          direct tensor transport
      add_binary_stream_handler  ← peer side (async handler)
      call_binary_stream_handler ← client side (returns reader/writer)
      P2P.send_raw_data / receive_raw_data   ← framed bytes I/O

Serialisation
-------------
  tensor → float32 numpy → raw bytes  (prefixed with ndim + shape via struct)
  bytes  → numpy frombuffer → torch tensor

Run
---
  conda activate soARM
  python test_peers.py
"""

import asyncio
import json
import struct
import time

import numpy as np
import torch
import torch.nn as nn
import hivemind
from hivemind.p2p import P2P
from hivemind.p2p.p2p_daemon_bindings.datastructures import PeerID
from hivemind.utils import get_dht_time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HIDDEN_SIZE = 64
SEQ_LEN = 8
HANDLER_NAME = "tensor_forward"
DHT_TTL = 300.0  # seconds before DHT entries expire

PEER0_DHT_PORT = 43300  # DHT discovery port for peer 0 (bootstrap)
PEER1_DHT_PORT = 43301  # DHT discovery port for peer 1
PEER0_P2P_PORT = 44211  # P2P tensor-transport port for peer 0
PEER1_P2P_PORT = 44212  # P2P tensor-transport port for peer 1

# ---------------------------------------------------------------------------
# Tensor serialisation
# ---------------------------------------------------------------------------


def tensor_to_bytes(t: torch.Tensor) -> bytes:
    """
    Encode a float32 tensor as raw bytes.

    Wire format:
        4 bytes  — ndim  (big-endian int32)
        4*ndim   — each dim (big-endian int32)
        remaining — float32 data in C order
    """
    arr = t.detach().cpu().float().numpy()
    header = struct.pack(">" + "i" * (1 + arr.ndim), arr.ndim, *arr.shape)
    return header + arr.tobytes()


def bytes_to_tensor(data: bytes) -> torch.Tensor:
    """Inverse of tensor_to_bytes."""
    ndim = struct.unpack(">i", data[:4])[0]
    shape = struct.unpack(">" + "i" * ndim, data[4 : 4 + ndim * 4])
    payload = data[4 + ndim * 4 :]
    arr = np.frombuffer(payload, dtype=np.float32).reshape(shape)
    return torch.from_numpy(arr.copy())


# ---------------------------------------------------------------------------
# Fake modules (stand-ins for transformer layer blocks)
# ---------------------------------------------------------------------------


class AddConstant(nn.Module):
    """Adds a fixed value to every tensor element — easy to assert."""

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"    [Module +{self.value}]  in shape {tuple(x.shape)}")
        return x + self.value


# ---------------------------------------------------------------------------
# Main — single event loop owns everything
# ---------------------------------------------------------------------------


async def main():
    print("=" * 60)
    print("hivemind raw P2P tensor test")
    print(f"  hidden_size={HIDDEN_SIZE}  seq_len={SEQ_LEN}")
    print("=" * 60)

    # ── 1. Bootstrap DHT ────────────────────────────────────────────────────
    # hivemind.DHT manages its own background process; start=True is sync.
    print("\n[1] Starting peer 0 DHT (bootstrap) ...")
    dht0 = hivemind.DHT(
        host_maddrs=[f"/ip4/127.0.0.1/tcp/{PEER0_DHT_PORT}"],
        start=True,
    )
    await asyncio.sleep(1.0)
    peer0_dht_maddr = str(dht0.get_visible_maddrs()[0])
    print(f"    Boot address: {peer0_dht_maddr}")

    # ── 2. Peer 1 DHT joins the network ─────────────────────────────────────
    print("\n[2] Starting peer 1 DHT ...")
    dht1 = hivemind.DHT(
        host_maddrs=[f"/ip4/127.0.0.1/tcp/{PEER1_DHT_PORT}"],
        initial_peers=[peer0_dht_maddr],
        start=True,
    )
    await asyncio.sleep(0.5)

    # ── 3. Peer 0 P2P node + handler ────────────────────────────────────────
    # P2P.create and add_binary_stream_handler are both async — await them
    # inside this loop so the handler is actually registered before the client
    # tries to connect.
    print("\n[3] Launching peer 0 P2P handler (layers 0-16, adds +1.0) ...")
    module0 = AddConstant(1.0)
    p2p0 = await P2P.create(host_maddrs=[f"/ip4/127.0.0.1/tcp/{PEER0_P2P_PORT}"])

    async def handle0(stream_info, reader, writer):
        data = await P2P.receive_raw_data(reader)
        with torch.no_grad():
            result = module0(bytes_to_tensor(data))
        await P2P.send_raw_data(tensor_to_bytes(result), writer)
        writer.close()

    await p2p0.add_binary_stream_handler(HANDLER_NAME, handle0)

    maddrs0 = await p2p0.get_visible_maddrs()
    maddr0 = str(maddrs0[0])
    info0 = json.dumps(
        {"maddr": maddr0, "peer_id": str(p2p0.peer_id), "layers": "0-16"}
    )
    dht0.store("peer0", info0, expiration_time=get_dht_time() + DHT_TTL)
    print(f"    Peer 0 registered — {maddr0}")

    # ── 4. Peer 1 P2P node + handler ────────────────────────────────────────
    print("\n[4] Launching peer 1 P2P handler (layers 17-32, adds +10.0) ...")
    module1 = AddConstant(10.0)
    p2p1 = await P2P.create(host_maddrs=[f"/ip4/127.0.0.1/tcp/{PEER1_P2P_PORT}"])

    async def handle1(stream_info, reader, writer):
        data = await P2P.receive_raw_data(reader)
        with torch.no_grad():
            result = module1(bytes_to_tensor(data))
        await P2P.send_raw_data(tensor_to_bytes(result), writer)
        writer.close()

    await p2p1.add_binary_stream_handler(HANDLER_NAME, handle1)

    maddrs1 = await p2p1.get_visible_maddrs()
    maddr1 = str(maddrs1[0])
    info1 = json.dumps(
        {"maddr": maddr1, "peer_id": str(p2p1.peer_id), "layers": "17-32"}
    )
    dht1.store("peer1", info1, expiration_time=get_dht_time() + DHT_TTL)
    print(f"    Peer 1 registered — {maddr1}")

    # ── 5. Client DHT + P2P ─────────────────────────────────────────────────
    print("\n[5] Starting client ...")
    client_dht = hivemind.DHT(initial_peers=[peer0_dht_maddr], start=True)
    await asyncio.sleep(1.0)

    # Discover peers from DHT
    def _get_peer(dht: hivemind.DHT, key: str, retries: int = 10) -> dict:
        for _ in range(retries):
            r = dht.get(key, latest=True)
            if r is not None:
                return json.loads(r.value)
            time.sleep(0.3)
        raise RuntimeError(f"Peer '{key}' not found in DHT")

    disc0 = _get_peer(client_dht, "peer0")
    disc1 = _get_peer(client_dht, "peer1")
    print(f"    Discovered peer0: layers={disc0['layers']}")
    print(f"    Discovered peer1: layers={disc1['layers']}")

    client_p2p = await P2P.create(
        initial_peers=[disc0["maddr"], disc1["maddr"]],
        host_maddrs=["/ip4/127.0.0.1/tcp/0"],
    )
    peer_id0 = PeerID.from_base58(disc0["peer_id"])
    peer_id1 = PeerID.from_base58(disc1["peer_id"])

    # ── 6. Send tensor through peer 0 ───────────────────────────────────────
    print("\n[6] Client → peer 0 ...")
    x = torch.zeros(SEQ_LEN, HIDDEN_SIZE)
    print(f"    Input: shape={tuple(x.shape)}  mean={x.mean():.2f}")

    _, r0, w0 = await client_p2p.call_binary_stream_handler(peer_id0, HANDLER_NAME)
    await P2P.send_raw_data(tensor_to_bytes(x), w0)
    y = bytes_to_tensor(await P2P.receive_raw_data(r0))
    w0.close()
    print(f"    After peer 0: mean={y.mean():.4f}  (expected 1.0)")
    assert abs(y.mean().item() - 1.0) < 1e-4, f"peer0 wrong: got {y.mean():.4f}"

    # ── 7. Forward to peer 1 ────────────────────────────────────────────────
    print("\n[7] Client → peer 1 ...")
    _, r1, w1 = await client_p2p.call_binary_stream_handler(peer_id1, HANDLER_NAME)
    await P2P.send_raw_data(tensor_to_bytes(y), w1)
    z = bytes_to_tensor(await P2P.receive_raw_data(r1))
    w1.close()
    print(f"    After peer 1: mean={z.mean():.4f}  (expected 11.0)")
    assert abs(z.mean().item() - 11.0) < 1e-4, f"peer1 wrong: got {z.mean():.4f}"

    # ── 8. Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print(f"  zeros → peer0(+1.0) → peer1(+10.0) → mean={z.mean():.2f}")
    print("  DHT discovery, P2P transport, tensor serialisation all work.")
    print("=" * 60)

    # ── 9. Cleanup ──────────────────────────────────────────────────────────
    print("\nShutting down ...")
    await client_p2p.shutdown()
    client_dht.shutdown()
    await p2p1.shutdown()
    await p2p0.shutdown()
    dht1.shutdown()
    dht0.shutdown()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
