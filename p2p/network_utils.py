"""
network_utils.py  (hivemind edition)
-------------------------------------
Helper functions for:
  - loading the peers.json config (now DHT bootstrap addresses + expert UIDs)
  - routing hidden states through remote experts via hivemind's DHT + gRPC
  - optional activation obfuscation (random orthogonal matrix)

What changed from the FastAPI version
--------------------------------------
Before: we serialised tensors to JSON lists and did HTTP POST to each peer.
Now:    hivemind handles all serialisation (protobuf) and transport (gRPC over
        libp2p) internally.  We just call expert(hidden) like a local nn.Module.

Key hivemind concepts used here
---------------------------------
hivemind.DHT
    A node in the distributed hash table.  Every process (client and peer)
    runs one.  Nodes find each other via an initial bootstrap address.
    Think of it as a decentralised service registry — no central server needed.

hivemind.RemoteExpert
    An nn.Module whose .forward() transparently calls a remote ExpertBackend
    over gRPC.  The DHT is used to locate the server that owns the expert UID.
    Tensor serialisation (protobuf + optional bitsandbytes compression) is
    handled automatically.
"""

import json
import torch
from typing import List, Tuple
import hivemind


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(path: str = "peers.json") -> dict:
    """
    Load the hivemind config.

    Expected format::

        {
            "initial_peers": ["/ip4/127.0.0.1/tcp/43211"],
            "experts":        ["layers_8_15", "layers_16_23"]
        }

    ``initial_peers`` is the multiaddress of the first DHT node (typically
    peer 0).  All other peers and the client pass it in their DHT constructor
    so they can join the same overlay network.

    ``experts`` lists expert UIDs in the order they should be called during
    inference.
    """
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_through_experts(
    hidden: torch.Tensor,
    expert_uids: List[str],
    dht: hivemind.DHT,
    obfuscate: bool = False,
) -> torch.Tensor:
    """
    Forward *hidden* sequentially through every expert in *expert_uids*.

    Each expert is a ``hivemind.RemoteExpert`` — it looks up the serving peer
    via the DHT and calls its forward pass over gRPC.  From the caller's
    perspective it is just ``expert(tensor)``.

    The tensor is reshaped before each call because hivemind's ExpertBackend
    treats the first dimension as the batch axis:

        client tensor : (1, seq_len, hidden_size)
        sent to expert: (seq_len, hidden_size)   ← seq acts as batch
        received back : (seq_len, hidden_size)
        restored to   : (1, seq_len, hidden_size)

    Obfuscation (optional)
    ~~~~~~~~~~~~~~~~~~~~~~
    Before each hop:  h_sent = h @ R
    After each hop:   h      = h_recv @ R^T
    where R is a random orthogonal matrix (R^{-1} = R^T).
    The peer computes on rotated activations and never sees the true values.
    """
    # Strip the batch-of-1 dimension for hivemind (batch=1 is the only
    # supported case in this prototype).
    assert hidden.shape[0] == 1, "Only batch_size=1 is supported"
    h = hidden.squeeze(0)  # (seq_len, hidden_size)

    for i, uid in enumerate(expert_uids):
        tag = f"expert '{uid}' (peer {i+1})"

        expert = hivemind.RemoteExpert(uid=uid, dht=dht)

        if obfuscate:
            R = _random_orthogonal(h.shape[-1], device=h.device)
            h_send = h @ R
        else:
            h_send = h

        print(f"Sending tensor to {tag}")
        h_recv = expert(h_send)  # gRPC call — hivemind serialises/deserialises
        print(f"{tag.capitalize()} finished")

        if obfuscate:
            h = h_recv @ R.T  # invert obfuscation
        else:
            h = h_recv

    return h.unsqueeze(0)  # restore (1, seq_len, hidden_size)


# ---------------------------------------------------------------------------
# Activation obfuscation helpers
# ---------------------------------------------------------------------------


def _random_orthogonal(dim: int, device: torch.device) -> torch.Tensor:
    """Return a random orthogonal (dim, dim) matrix via QR decomposition."""
    G = torch.randn(dim, dim, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(G)
    return Q
