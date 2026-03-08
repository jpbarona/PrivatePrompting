import asyncio
import argparse
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from protocol import (
    KIND_ERROR,
    KIND_SHUTDOWN,
    KIND_TENSOR,
    KIND_HELLO,
    KIND_READY,
    decode_frame,
    encode_frame,
)
from p2p_transport import call_handler, create_p2p_node, discover_worker, join_dht


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--prompt", default="Hey! How are you feeling today?")
    p.add_argument("--num-new-tokens", type=int, default=16)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--num-middle-partitions", type=int, default=2)
    p.add_argument("--host-ip", required=True)
    p.add_argument("--dht-port", type=int, required=True)
    p.add_argument("--bootstrap-maddr", required=True)
    p.add_argument("--w1-dht-key", default="inference_w1")
    p.add_argument("--w2-dht-key", default="inference_w2")
    p.add_argument("--max-seq", type=int, default=2048)
    return p.parse_args()


def run_layers(hidden_states, layers, rotary_emb, device):
    seq_len = hidden_states.shape[1]
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    position_embeddings = rotary_emb(hidden_states, position_ids)
    causal_mask = torch.triu(
        torch.full(
            (seq_len, seq_len),
            torch.finfo(hidden_states.dtype).min,
            device=device,
            dtype=hidden_states.dtype,
        ),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)
    for layer in layers:
        hidden_states = layer(
            hidden_states,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            use_cache=False,
        )
    return hidden_states


def get_baseline(prompt, num_tokens, model, tokenizer, device):
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inp["input_ids"].clone()
    with torch.no_grad():
        for _ in range(num_tokens):
            out = model(input_ids=input_ids, use_cache=False)
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=-1)
    return input_ids[0, inp["input_ids"].shape[1] :].tolist()


async def run_chain_inference(
    *,
    p2p,
    w1_info,
    prompt,
    num_tokens,
    tokenizer,
    embed_module,
    first_k_layers,
    last_k_layers,
    final_norm,
    lm_head_module,
    rotary_emb,
    device,
    max_nbytes,
):
    async def send_recv(hidden_states):
        t = hidden_states.cpu().float()
        nbytes = t.numel() * t.element_size()
        if nbytes > max_nbytes:
            raise ValueError(f"nbytes={nbytes} > max_nbytes={max_nbytes}")
        request = encode_frame(
            {"kind": KIND_TENSOR, "shape": tuple(t.shape), "nbytes": nbytes},
            t.numpy().tobytes(),
        )
        response = await call_handler(
            p2p=p2p,
            peer_id=w1_info.peer_id,
            peer_maddr=w1_info.maddr,
            handler_name=w1_info.handler_name,
            payload_bytes=request,
        )
        frame, out_bytes = decode_frame(response, max_nbytes)
        kind = frame["kind"]
        if kind == KIND_ERROR:
            raise RuntimeError(f"Worker error: {frame['message']}")
        if kind != KIND_TENSOR:
            raise RuntimeError(f"Unexpected response frame: {frame}")
        out_shape = tuple(frame["shape"])
        return torch.frombuffer(out_bytes, dtype=torch.float32).reshape(out_shape).clone().to(device)

    inp = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inp["input_ids"].clone()
    with torch.no_grad():
        for _ in range(num_tokens):
            hidden = embed_module(input_ids)
            hidden = run_layers(hidden, first_k_layers, rotary_emb, device)
            hidden = await send_recv(hidden)
            hidden = run_layers(hidden, last_k_layers, rotary_emb, device)
            hidden = final_norm(hidden)
            logits = lm_head_module(hidden[:, -1:, :])
            next_id = logits.argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_id], dim=-1)
    return input_ids[0, inp["input_ids"].shape[1] :].tolist()


async def main():
    args = parse_args()
    if args.num_middle_partitions != 2:
        raise ValueError("Phase 6 keeps the same topology as Phase 5 and requires exactly 2 middle partitions.")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model: Any = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    model = model.to(device=device)
    model.eval()

    base = getattr(model, "model", model)
    embed_module = getattr(base, "embed_tokens", None) or getattr(base, "embed", None)
    layers_list = getattr(base, "layers", None)
    final_norm = getattr(base, "norm", None)
    lm_head_module = getattr(model, "lm_head", None)
    rotary_emb = getattr(base, "rotary_emb", None)
    if not all([embed_module is not None, layers_list is not None, final_norm is not None, lm_head_module is not None, rotary_emb is not None]):
        raise RuntimeError("Model does not expose expected modules.")
    assert layers_list is not None
    assert embed_module is not None
    assert final_norm is not None
    assert lm_head_module is not None
    assert rotary_emb is not None

    num_blocks = len(layers_list)
    first_k_layers = layers_list[: args.k]
    last_k_layers = layers_list[-args.k :]
    middle_start = args.k
    middle_end = num_blocks - args.k
    middle_count = middle_end - middle_start
    if middle_count <= 0 or middle_count % args.num_middle_partitions != 0:
        raise ValueError("Invalid middle partitioning.")
    chunk = middle_count // args.num_middle_partitions
    ranges = [(middle_start + i * chunk, middle_start + (i + 1) * chunk) for i in range(args.num_middle_partitions)]

    hidden_size = model.config.hidden_size
    batch_size = 1
    max_nbytes = batch_size * args.max_seq * hidden_size * 4

    print(f"total blocks={num_blocks}, first_k={args.k}, middle_count={middle_count}, last_k={args.k}")
    print(f"middle partition ranges={ranges}")
    print(f"device={device}, max_nbytes={max_nbytes}")

    dht, dht_maddr = join_dht(
        dht_port=args.dht_port,
        bootstrap_maddr=args.bootstrap_maddr,
        host_ip=args.host_ip,
    )
    p2p = await create_p2p_node(p2p_port=0)
    w1_info = discover_worker(dht, key=args.w1_dht_key)
    w2_info = discover_worker(dht, key=args.w2_dht_key)
    if w1_info.layer_start != ranges[0][0] or w1_info.layer_end != ranges[0][1]:
        raise RuntimeError(
            f"W1 layers mismatch: discovered [{w1_info.layer_start},{w1_info.layer_end}), expected {ranges[0]}"
        )
    if w2_info.layer_start != ranges[1][0] or w2_info.layer_end != ranges[1][1]:
        raise RuntimeError(
            f"W2 layers mismatch: discovered [{w2_info.layer_start},{w2_info.layer_end}), expected {ranges[1]}"
        )
    print(f"parent dht: {dht_maddr}")
    print(f"discovered w1: {w1_info.maddr}")
    print(f"discovered w2: {w2_info.maddr}")

    handshake_ok = False
    primary_exc = None
    try:
        hello_response = await call_handler(
            p2p=p2p,
            peer_id=w1_info.peer_id,
            peer_maddr=w1_info.maddr,
            handler_name=w1_info.handler_name,
            payload_bytes=encode_frame({"kind": KIND_HELLO, "role": "parent"}),
        )
        hello_frame, _ = decode_frame(hello_response, max_nbytes)
        if hello_frame["kind"] != KIND_READY:
            raise RuntimeError(f"Expected READY, got {hello_frame}")
        handshake_ok = True

        baseline = get_baseline(args.prompt, args.num_new_tokens, model, tokenizer, device)
        split = await run_chain_inference(
            p2p=p2p,
            w1_info=w1_info,
            prompt=args.prompt,
            num_tokens=args.num_new_tokens,
            tokenizer=tokenizer,
            embed_module=embed_module,
            first_k_layers=first_k_layers,
            last_k_layers=last_k_layers,
            final_norm=final_norm,
            lm_head_module=lm_head_module,
            rotary_emb=rotary_emb,
            device=device,
            max_nbytes=max_nbytes,
        )

        print("baseline ids:", baseline)
        print("split ids:   ", split)
        print("match:", baseline == split)
        print("baseline decoded:", tokenizer.decode(baseline, skip_special_tokens=True))
        print("split decoded:   ", tokenizer.decode(split, skip_special_tokens=True))
    except BaseException as exc:
        primary_exc = exc
        raise
    finally:
        if handshake_ok:
            try:
                shutdown_response = await call_handler(
                    p2p=p2p,
                    peer_id=w1_info.peer_id,
                    peer_maddr=w1_info.maddr,
                    handler_name=w1_info.handler_name,
                    payload_bytes=encode_frame({"kind": KIND_SHUTDOWN}),
                )
                shutdown_frame, _ = decode_frame(shutdown_response, max_nbytes)
                if shutdown_frame["kind"] != KIND_SHUTDOWN:
                    raise RuntimeError(f"Unexpected final frame: {shutdown_frame}")
            except Exception as shutdown_exc:
                if primary_exc is None:
                    raise
                print(f"shutdown warning: {type(shutdown_exc).__name__}: {shutdown_exc}")
        await p2p.shutdown()
        dht.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
