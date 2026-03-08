"""
Thin FastAPI wrapper around the split-inference pipeline.
Exposes a single POST /infer endpoint for the frontend.

Usage:
    python api.py \
        --host-ip 127.0.0.1 \
        --dht-port 44100 \
        --bootstrap-maddr <maddr> \
        [--model-name Qwen/Qwen2.5-0.5B-Instruct] \
        [--k 2] \
        [--w1-dht-key inference_w1] \
        [--w2-dht-key inference_w2] \
        [--api-port 8000]
"""

import argparse
import asyncio
from contextlib import asynccontextmanager
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from p2p_transport import call_handler, create_p2p_node, discover_worker, join_dht
from parent_client import run_chain_inference
from protocol import KIND_HELLO, KIND_READY, decode_frame, encode_frame


# ── CLI args ──────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--num-middle-partitions", type=int, default=2)
    p.add_argument("--host-ip", required=True)
    p.add_argument("--dht-port", type=int, required=True)
    p.add_argument("--bootstrap-maddr", required=True)
    p.add_argument("--w1-dht-key", default="inference_w1")
    p.add_argument("--w2-dht-key", default="inference_w2")
    p.add_argument(
        "--run-id",
        default="",
        help="Optional suffix appended to DHT keys (matches e2e --run-id)",
    )
    p.add_argument("--max-seq", type=int, default=2048)
    p.add_argument("--api-port", type=int, default=8000)
    p.add_argument("--api-host", default="0.0.0.0")
    return p.parse_args()


args = parse_args()

# ── Shared state loaded once at startup ───────────────────────────────────────

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32

    print(f"[api] Loading model {args.model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model: Any = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype
    )
    model = model.to(device=device)
    model.eval()

    base = getattr(model, "model", model)
    embed_module = getattr(base, "embed_tokens", None) or getattr(base, "embed", None)
    layers_list = getattr(base, "layers", None)
    final_norm = getattr(base, "norm", None)
    lm_head_module = getattr(model, "lm_head", None)
    rotary_emb = getattr(base, "rotary_emb", None)

    if not all([embed_module, layers_list, final_norm, lm_head_module, rotary_emb]):
        raise RuntimeError("Model does not expose expected modules.")

    num_blocks = len(layers_list)
    first_k = layers_list[: args.k]
    last_k = layers_list[-args.k :]
    middle_count = num_blocks - 2 * args.k
    chunk = middle_count // args.num_middle_partitions
    ranges = [
        (args.k + i * chunk, args.k + (i + 1) * chunk)
        for i in range(args.num_middle_partitions)
    ]
    hidden_size = model.config.hidden_size
    max_nbytes = 1 * args.max_seq * hidden_size * 4

    print(f"[api] Joining DHT on port {args.dht_port} …")
    dht, dht_maddr = join_dht(
        dht_port=args.dht_port,
        bootstrap_maddr=args.bootstrap_maddr,
        host_ip=args.host_ip,
    )

    print("[api] Creating P2P node …")
    p2p = await create_p2p_node(p2p_port=0)

    def _dht_key(base: str) -> str:
        return f"{base}_{args.run_id}" if args.run_id else base

    print("[api] Discovering workers …")
    w1_info = discover_worker(dht, key=_dht_key(args.w1_dht_key))
    w2_info = discover_worker(dht, key=_dht_key(args.w2_dht_key))
    print(f"[api] w1: {w1_info.maddr}  w2: {w2_info.maddr}")

    if w1_info.layer_start != ranges[0][0] or w1_info.layer_end != ranges[0][1]:
        raise RuntimeError(
            f"W1 layers mismatch: {w1_info.layer_start}-{w1_info.layer_end} vs {ranges[0]}"
        )
    if w2_info.layer_start != ranges[1][0] or w2_info.layer_end != ranges[1][1]:
        raise RuntimeError(
            f"W2 layers mismatch: {w2_info.layer_start}-{w2_info.layer_end} vs {ranges[1]}"
        )

    state.update(
        device=device,
        tokenizer=tokenizer,
        embed_module=embed_module,
        first_k=first_k,
        last_k=last_k,
        final_norm=final_norm,
        lm_head_module=lm_head_module,
        rotary_emb=rotary_emb,
        p2p=p2p,
        w1_info=w1_info,
        max_nbytes=max_nbytes,
        dht=dht,
    )
    # ── HELLO handshake: warm up w1→w2 connection (matches parent_client flow) ──
    print("[api] Sending HELLO handshake …")
    hello_response = await call_handler(
        p2p=p2p,
        peer_id=w1_info.peer_id,
        peer_maddr=w1_info.maddr,
        handler_name=w1_info.handler_name,
        payload_bytes=encode_frame({"kind": KIND_HELLO, "role": "parent"}),
    )
    hello_frame, _ = decode_frame(hello_response, max_nbytes)
    if hello_frame["kind"] != KIND_READY:
        raise RuntimeError(f"Expected READY from w1, got {hello_frame}")
    print("[api] Ready.")

    yield  # ── server is running ──

    # Workers are persistent (non-dying) — do NOT send KIND_SHUTDOWN
    await p2p.shutdown()
    dht.shutdown()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. "
    "Answer the user's question directly and clearly."
)


class InferRequest(BaseModel):
    prompt: str


class InferResponse(BaseModel):
    response: str


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=422, detail="prompt must not be empty")
    formatted = state["tokenizer"].apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    try:
        token_ids = await run_chain_inference(
            p2p=state["p2p"],
            w1_info=state["w1_info"],
            prompt=formatted,
            num_tokens=512,  # ceiling — loop breaks early at EOS
            tokenizer=state["tokenizer"],
            embed_module=state["embed_module"],
            first_k_layers=state["first_k"],
            last_k_layers=state["last_k"],
            final_norm=state["final_norm"],
            lm_head_module=state["lm_head_module"],
            rotary_emb=state["rotary_emb"],
            device=state["device"],
            max_nbytes=state["max_nbytes"],
        )
        text = state["tokenizer"].decode(token_ids, skip_special_tokens=True)
        return InferResponse(response=text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(app, host=args.api_host, port=args.api_port)
