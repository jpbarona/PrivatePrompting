import argparse
import socket
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from protocol import (
    KIND_ERROR,
    KIND_SHUTDOWN,
    KIND_TENSOR,
    configure_sock,
    expect_hello_send_ready,
    recv_frame,
    send_frame,
    send_hello_expect_ready,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--prompt", default="Hey! How are you feeling today?")
    p.add_argument("--num-new-tokens", type=int, default=16)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--num-middle-partitions", type=int, default=2)
    p.add_argument("--port-parent", type=int, default=5100)
    p.add_argument("--port-w1", type=int, default=5101)
    p.add_argument("--max-seq", type=int, default=2048)
    return p.parse_args()


def connect_parent_to_w1_with_retry(port_w1, timeout_s=120.0):
    deadline = time.monotonic() + timeout_s
    sleep_s = 0.05
    last_error = None
    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        configure_sock(sock)
        try:
            sock.connect(("127.0.0.1", port_w1))
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
            time.sleep(sleep_s)
            sleep_s = min(1.0, sleep_s * 2)
    raise TimeoutError(f"Timed out connecting parent->W1; last_error={last_error}")


def accept_w2(listen_sock, timeout_s=120.0):
    deadline = time.monotonic() + timeout_s
    listen_sock.settimeout(1.0)
    while time.monotonic() < deadline:
        try:
            sock, _ = listen_sock.accept()
            configure_sock(sock)
            return sock
        except socket.timeout:
            continue
    raise TimeoutError("Timed out waiting for W2->parent connection")


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


def run_chain_inference(
    sock_to_w1,
    sock_from_w2,
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
    def send_recv(hidden_states):
        t = hidden_states.cpu().float()
        nbytes = t.numel() * t.element_size()
        if nbytes > max_nbytes:
            raise ValueError(f"nbytes={nbytes} > max_nbytes={max_nbytes}")
        send_frame(
            sock_to_w1,
            {"kind": KIND_TENSOR, "shape": tuple(t.shape), "nbytes": nbytes},
            t.numpy().tobytes(),
        )
        frame, out_bytes = recv_frame(sock_from_w2, max_nbytes)
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
            hidden = send_recv(hidden)
            hidden = run_layers(hidden, last_k_layers, rotary_emb, device)
            hidden = final_norm(hidden)
            logits = lm_head_module(hidden[:, -1:, :])
            next_id = logits.argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_id], dim=-1)
    return input_ids[0, inp["input_ids"].shape[1] :].tolist()


def main():
    args = parse_args()
    if args.num_middle_partitions != 2:
        raise ValueError("Phase 6 keeps the same topology as Phase 5 and requires exactly 2 middle partitions.")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype).to(device)
    model.eval()

    base = getattr(model, "model", model)
    embed_module = getattr(base, "embed_tokens", None) or getattr(base, "embed", None)
    layers_list = getattr(base, "layers", None)
    final_norm = getattr(base, "norm", None)
    lm_head_module = getattr(model, "lm_head", None)
    rotary_emb = getattr(base, "rotary_emb", None)
    if not all([embed_module is not None, layers_list is not None, final_norm is not None, lm_head_module is not None, rotary_emb is not None]):
        raise RuntimeError("Model does not expose expected modules.")

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

    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_sock.bind(("", args.port_parent))
    listen_sock.listen(1)

    sock_to_w1 = None
    sock_from_w2 = None
    try:
        sock_to_w1 = connect_parent_to_w1_with_retry(args.port_w1)
        sock_from_w2 = accept_w2(listen_sock)
        expect_hello_send_ready(sock_from_w2, role="parent", max_nbytes=max_nbytes)
        send_hello_expect_ready(sock_to_w1, role="parent", max_nbytes=max_nbytes)

        baseline = get_baseline(args.prompt, args.num_new_tokens, model, tokenizer, device)
        split = run_chain_inference(
            sock_to_w1=sock_to_w1,
            sock_from_w2=sock_from_w2,
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
    finally:
        to_raise = None
        if sock_to_w1 is not None:
            try:
                send_frame(sock_to_w1, {"kind": KIND_SHUTDOWN})
            except Exception:
                pass
        if sock_from_w2 is not None:
            try:
                frame, _ = recv_frame(sock_from_w2, max_nbytes)
                if frame["kind"] != KIND_SHUTDOWN:
                    to_raise = RuntimeError(f"Unexpected final frame: {frame}")
            except Exception as e:
                to_raise = e
        if sock_to_w1 is not None:
            sock_to_w1.close()
        if sock_from_w2 is not None:
            sock_from_w2.close()
        listen_sock.close()
        if to_raise is not None:
            raise to_raise


if __name__ == "__main__":
    main()
