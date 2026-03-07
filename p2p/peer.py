"""
peer.py  (hivemind edition)
----------------------------
Each peer runs a hivemind.Server that:
  1. Joins the DHT overlay (the decentralised peer registry).
  2. Registers a named ExpertBackend that wraps its assigned transformer layers.
  3. Serves forward-pass requests over gRPC (no HTTP, no JSON).

How it works
-------------
hivemind.DHT
    A lightweight distributed hash table node built on libp2p.  Every process
    in the network (peers AND the client) runs one.  They discover each other
    via a bootstrap address passed with --initial-peer.  The very first peer
    starts fresh (no --initial-peer) and prints its own address so everyone
    else can bootstrap from it.

hivemind.ExpertBackend
    Wraps an nn.Module so that hivemind knows its input/output tensor schemas.
    The Server registers one or more backends with the DHT under their UIDs.

hivemind.Server
    Spawns worker processes that receive protobuf-encoded tensors over gRPC,
    run the wrapped module, and return the result.  All network + serialisation
    code lives inside hivemind — we just provide the PyTorch layer logic.

LayerBlock.forward signature
-----------------------------
ExpertBackend treats the first tensor dimension as the batch axis.  Because
batch_size=1 throughout this prototype, we let the token sequence occupy that
dimension:

    received  : (seq_len, hidden_size)   ← seq_len acts as hivemind's "batch"
    processed : (1, seq_len, hidden_size) internally
    returned  : (seq_len, hidden_size)

Usage
-----
    # First peer — starts the DHT, prints its multiaddress
    python peer.py --layers 8-15 --port 43211

    # Second peer — joins the existing DHT
    python peer.py --layers 16-23 --port 43212 \\
        --initial-peer /ip4/127.0.0.1/tcp/43211
"""

import argparse
import logging
import time

import torch
import torch.nn as nn
import hivemind
from transformers import AutoModelForCausalLM

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Hivemind peer node")
parser.add_argument("--layers", type=str, required=True, help="e.g. '8-15'")
parser.add_argument("--port", type=int, default=43211, help="DHT TCP port")
parser.add_argument(
    "--initial-peer",
    type=str,
    default=None,
    help="Multiaddress of an existing DHT node to bootstrap from",
)
parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B")
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()

layer_start, layer_end = [int(x) for x in args.layers.split("-")]
DEVICE = args.device
EXPERT_UID = f"layers_{layer_start}_{layer_end}"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layer block  (the nn.Module that peers actually execute)
# ---------------------------------------------------------------------------


class LayerBlock(nn.Module):
    """
    Wraps a contiguous slice of transformer decoder layers.

    Input/output shape seen by hivemind: (seq_len, hidden_size)
    We unsqueeze/squeeze the batch=1 dimension internally.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        layer_start: int,
        layer_end: int,
        port: int,
        device: str,
    ):
        super().__init__()
        self.layers = layers
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.port = port
        self.device = device

    def forward(self, hidden_2d: torch.Tensor) -> torch.Tensor:
        """
        hidden_2d: (seq_len, hidden_size)  — seq_len is hivemind's batch axis.
        """
        logger.info(
            f"[Peer {self.port}] Received tensor shape {tuple(hidden_2d.shape)}"
        )
        logger.info(
            f"[Peer {self.port}] Running layers {self.layer_start}-{self.layer_end}"
        )

        seq_len = hidden_2d.shape[0]
        hidden = hidden_2d.unsqueeze(0).to(self.device)  # (1, seq_len, hidden_size)
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(
            0
        )  # (1, seq_len)

        with torch.no_grad():
            for layer in self.layers:
                hidden = layer(
                    hidden,
                    attention_mask=None,
                    position_ids=position_ids,
                )[0]

        logger.info(f"[Peer {self.port}] Returning tensor")
        return hidden.squeeze(0).cpu()  # (seq_len, hidden_size)


# ---------------------------------------------------------------------------
# Load model layers
# ---------------------------------------------------------------------------

logger.info(
    f"[Peer {args.port}] Loading {args.model} layers {layer_start}-{layer_end} ..."
)
base_model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True
)
layers = nn.ModuleList(
    [base_model.model.layers[i] for i in range(layer_start, layer_end + 1)]
).to(DEVICE)
del base_model
logger.info(f"[Peer {args.port}] Layers {layer_start}-{layer_end} ready")

hidden_size = layers[0].self_attn.q_proj.in_features

# ---------------------------------------------------------------------------
# Build the hivemind ExpertBackend
# ---------------------------------------------------------------------------
# args_schema describes the input tensor shape *per sample*.
# We use (hidden_size,) so hivemind knows each sample in the "batch"
# (= each token) is a 1-D vector of size hidden_size.

module = LayerBlock(layers, layer_start, layer_end, args.port, DEVICE)

backend = hivemind.ExpertBackend(
    name=EXPERT_UID,
    module=module,
    optimizer=None,  # inference only — no optimiser needed
    args_schema=(hivemind.BatchTensorDescriptor(hidden_size),),
    outputs_schema=hivemind.BatchTensorDescriptor(hidden_size),
    max_batch_size=2048,  # max tokens per call (seq_len upper bound)
)

# ---------------------------------------------------------------------------
# Start the DHT node
# ---------------------------------------------------------------------------
# host_maddrs: the address this peer LISTENS on for incoming DHT traffic.
# initial_peers: bootstrap into an existing DHT (omit on the very first peer).

initial_peers = [args.initial_peer] if args.initial_peer else []

dht = hivemind.DHT(
    host_maddrs=[f"/ip4/0.0.0.0/tcp/{args.port}"],
    initial_peers=initial_peers,
    start=True,  # runs DHT in a background thread
)

# Print own visible addresses so other peers / the client can bootstrap.
visible = dht.get_visible_maddrs()
logger.info(f"[Peer {args.port}] DHT addresses: {visible}")
logger.info(
    f"[Peer {args.port}] To bootstrap from this peer, pass:\n"
    f"  --initial-peer {visible[0]}"
)

# ---------------------------------------------------------------------------
# Start the Server
# ---------------------------------------------------------------------------
# The Server registers EXPERT_UID in the DHT and starts gRPC workers.

server = hivemind.Server(
    dht,
    expert_backends={EXPERT_UID: backend},
    num_connection_handlers=4,
)

logger.info(f"[Peer {args.port}] Server starting for expert '{EXPERT_UID}'")
server.run_in_background(await_ready=True)
logger.info(f"[Peer {args.port}] Ready — serving '{EXPERT_UID}'")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    logger.info(f"[Peer {args.port}] Shutting down")
    server.shutdown()
    dht.shutdown()
