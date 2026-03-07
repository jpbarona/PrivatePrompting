"""
client.py  (hivemind edition)
------------------------------
Inference client for split-model execution using hivemind.

Differences from the FastAPI version
--------------------------------------
Before: peers.json held {host, port} pairs; client did HTTP POST to each.
Now:    peers.json holds DHT bootstrap address + expert UIDs.  The client:
          1. Joins the same DHT as the peers.
          2. Uses hivemind.RemoteExpert to call each peer's forward pass.
             RemoteExpert is an nn.Module — calling it looks like a local op.
          3. Everything else (local layers, norm, lm_head) is unchanged.

Usage
-----
    python client.py --prompt "Once upon a time"
    python client.py --prompt "Hello" --obfuscate
    python client.py --prompt "Hi" --local-prefix-end 7 --local-suffix-start 24
"""

import argparse
import logging

import torch
import hivemind
from transformers import AutoTokenizer, AutoModelForCausalLM

from network_utils import load_config, route_through_experts

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Split-model inference client (hivemind)")
parser.add_argument("--prompt", type=str, default="Once upon a time")
parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B")
parser.add_argument("--config", type=str, default="peers.json")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--local-prefix-end", type=int, default=7)
parser.add_argument("--local-suffix-start", type=int, default=24)
parser.add_argument("--max-new-tokens", type=int, default=20)
parser.add_argument(
    "--obfuscate",
    action="store_true",
    help="Apply random-orthogonal activation obfuscation before each peer hop",
)
args = parser.parse_args()

DEVICE = args.device

# ---------------------------------------------------------------------------
# Load the peers.json config and join the DHT
# ---------------------------------------------------------------------------

cfg = load_config(args.config)
initial_peers: list = cfg["initial_peers"]
expert_uids: list = cfg["experts"]

logger.info(f"Joining DHT via {initial_peers} ...")
dht = hivemind.DHT(
    initial_peers=initial_peers,
    start=True,  # background thread
)
logger.info(f"DHT joined.  Expert route: {expert_uids}")

# ---------------------------------------------------------------------------
# Load local model components
# ---------------------------------------------------------------------------

logger.info(f"Loading tokenizer and model: {args.model}")
tokenizer = AutoTokenizer.from_pretrained(args.model)
full_model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True
).to(DEVICE)
full_model.eval()

transformer = full_model.model
all_layers = transformer.layers
num_layers = len(all_layers)

prefix_end = args.local_prefix_end
suffix_start = min(args.local_suffix_start, num_layers)
client_prefix_layers = list(range(0, prefix_end + 1))
client_suffix_layers = list(range(suffix_start, num_layers))

logger.info(f"Client prefix layers : 0–{prefix_end}")
if client_suffix_layers:
    logger.info(f"Client suffix layers : {suffix_start}–{num_layers - 1}")
else:
    logger.info("Client suffix layers : none")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_layers(hidden: torch.Tensor, layer_indices: list) -> torch.Tensor:
    """Run a subset of local transformer layers."""
    batch, seq_len, _ = hidden.shape
    position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0).expand(batch, -1)
    with torch.no_grad():
        for idx in layer_indices:
            hidden = all_layers[idx](
                hidden, attention_mask=None, position_ids=position_ids
            )[0]
    return hidden


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------


def generate(prompt: str, max_new_tokens: int, obfuscate: bool) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    generated_ids = input_ids.clone()

    for step in range(max_new_tokens):
        with torch.no_grad():
            hidden = transformer.embed_tokens(generated_ids)

        # --- client prefix layers -------------------------------------------
        if client_prefix_layers:
            logger.info(f"Client running layers 0-{client_prefix_layers[-1]}")
            hidden = _run_layers(hidden, client_prefix_layers)

        # --- route through remote experts (hivemind) ------------------------
        if expert_uids:
            hidden = route_through_experts(
                hidden, expert_uids, dht=dht, obfuscate=obfuscate
            )

        # --- client suffix layers -------------------------------------------
        if client_suffix_layers:
            logger.info(
                f"Client running suffix layers "
                f"{client_suffix_layers[0]}-{client_suffix_layers[-1]}"
            )
            hidden = _run_layers(hidden, client_suffix_layers)

        # --- norm + lm_head → next token ------------------------------------
        with torch.no_grad():
            logits = full_model.lm_head(transformer.norm(hidden))

        logger.info("Client generating token")
        next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    new_ids = generated_ids[:, input_ids.shape[1] :]
    return tokenizer.decode(new_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info(f"Prompt: {args.prompt!r}")
    logger.info("-" * 50)
    result = generate(args.prompt, args.max_new_tokens, args.obfuscate)
    logger.info("-" * 50)
    print(f"\nGenerated: {result}")
    dht.shutdown()
