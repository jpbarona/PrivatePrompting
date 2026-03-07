"""
test_peers.py
--------------
Minimal connectivity test for the hivemind P2P layer.

Does NOT load Qwen or any real model.
Each "peer" runs a tiny nn.Module (adds a constant to the tensor) so you
can verify the tensor actually travelled through both peers and came back
with the right values.

What this tests
---------------
1. Two DHT nodes can form a network on localhost.
2. hivemind.Server can register and serve an ExpertBackend.
3. hivemind.RemoteExpert can find the server via the DHT and call it.
4. Tensors survive the round-trip (serialise → gRPC → deserialise).
5. Sequential routing works (peer1 output becomes peer2 input).

Run
---
    pip install hivemind torch
    python test_peers.py
"""

import time
import threading
import torch
import torch.nn as nn
import hivemind

# ---------------------------------------------------------------------------
# Fake "layer block" — just adds a known constant so we can assert the result
# ---------------------------------------------------------------------------

class AddConstant(nn.Module):
    """Adds `value` to every element. Makes the round-trip verifiable."""
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"  [AddConstant +{self.value}] received shape {tuple(x.shape)}")
        return x + self.value


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HIDDEN_SIZE = 64       # small, fast
SEQ_LEN     = 8        # 8 fake "tokens"

PEER0_PORT  = 44211    # also serves as the DHT bootstrap
PEER1_PORT  = 44212

EXPERT_UID_0 = "peer0_expert"
EXPERT_UID_1 = "peer1_expert"

PEER0_VALUE  = 1.0     # peer0 adds 1 to every element
PEER1_VALUE  = 10.0    # peer1 adds 10

# ---------------------------------------------------------------------------
# Helper — build a hivemind Server for one expert
# ---------------------------------------------------------------------------

def make_server(
    expert_uid: str,
    module: nn.Module,
    dht_port: int,
    initial_peers: list,
) -> tuple[hivemind.DHT, hivemind.Server]:
    """
    Start a DHT node + ExpertBackend Server in background threads.
    Returns (dht, server) so the caller can shut them down later.
    """
    backend = hivemind.ExpertBackend(
        name=expert_uid,
        module=module,
        optimizer=None,
        args_schema=(hivemind.BatchTensorDescriptor(HIDDEN_SIZE),),
        outputs_schema=hivemind.BatchTensorDescriptor(HIDDEN_SIZE),
        max_batch_size=512,
    )

    dht = hivemind.DHT(
        host_maddrs=[f"/ip4/127.0.0.1/tcp/{dht_port}"],
        initial_peers=initial_peers,
        start=True,
    )

    server = hivemind.Server(
        dht,
        expert_backends={expert_uid: backend},
        num_connection_handlers=2,
    )
    server.run_in_background(await_ready=True)

    return dht, server


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("hivemind peer connectivity test")
    print(f"  hidden_size={HIDDEN_SIZE}  seq_len={SEQ_LEN}")
    print("=" * 60)

    # ── 1. Start peer 0 (bootstraps the DHT) ────────────────────────────────
    print("\n[1] Starting peer 0 (DHT bootstrap) ...")
    dht0, server0 = make_server(
        EXPERT_UID_0,
        AddConstant(PEER0_VALUE),
        dht_port=PEER0_PORT,
        initial_peers=[],        # first node — no bootstrap needed
    )
    peer0_addr = str(dht0.get_visible_maddrs()[0])
    print(f"    peer 0 DHT address: {peer0_addr}")

    # ── 2. Start peer 1 (joins peer 0's DHT) ────────────────────────────────
    print("\n[2] Starting peer 1 (joins existing DHT) ...")
    dht1, server1 = make_server(
        EXPERT_UID_1,
        AddConstant(PEER1_VALUE),
        dht_port=PEER1_PORT,
        initial_peers=[peer0_addr],
    )
    print(f"    peer 1 DHT address: {dht1.get_visible_maddrs()[0]}")

    # Give the DHT a moment to propagate registrations
    time.sleep(1.0)

    # ── 3. Client joins the DHT ──────────────────────────────────────────────
    print("\n[3] Client joining DHT ...")
    client_dht = hivemind.DHT(
        initial_peers=[peer0_addr],
        start=True,
    )
    print("    Client DHT ready")

    # ── 4. Build a random input tensor ──────────────────────────────────────
    # Shape: (SEQ_LEN, HIDDEN_SIZE) — seq acts as hivemind's batch axis
    x = torch.zeros(SEQ_LEN, HIDDEN_SIZE)
    print(f"\n[4] Input tensor  : shape={tuple(x.shape)}  mean={x.mean():.2f}")

    # ── 5. Call peer 0 ──────────────────────────────────────────────────────
    print(f"\n[5] Calling {EXPERT_UID_0} (AddConstant +{PEER0_VALUE}) ...")
    expert0 = hivemind.RemoteExpert(uid=EXPERT_UID_0, dht=client_dht)
    y = expert0(x)
    print(f"    Result shape={tuple(y.shape)}  mean={y.mean():.2f}  (expected {PEER0_VALUE:.2f})")
    assert abs(y.mean().item() - PEER0_VALUE) < 1e-4, "peer 0 result incorrect!"

    # ── 6. Call peer 1 (with peer 0's output) ───────────────────────────────
    print(f"\n[6] Calling {EXPERT_UID_1} (AddConstant +{PEER1_VALUE}) ...")
    expert1 = hivemind.RemoteExpert(uid=EXPERT_UID_1, dht=client_dht)
    z = expert1(y)
    expected_final = PEER0_VALUE + PEER1_VALUE
    print(f"    Result shape={tuple(z.shape)}  mean={z.mean():.2f}  (expected {expected_final:.2f})")
    assert abs(z.mean().item() - expected_final) < 1e-4, "peer 1 result incorrect!"

    # ── 7. Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print(f"  0.0  →  peer0 (+{PEER0_VALUE})  →  {PEER0_VALUE}")
    print(f"  {PEER0_VALUE}  →  peer1 (+{PEER1_VALUE}) →  {z.mean():.2f}")
    print("  Tensor serialisation, DHT lookup and gRPC transport all work.")
    print("=" * 60)

    # ── 8. Cleanup ──────────────────────────────────────────────────────────
    client_dht.shutdown()
    server1.shutdown(); dht1.shutdown()
    server0.shutdown(); dht0.shutdown()


if __name__ == "__main__":
    main()
