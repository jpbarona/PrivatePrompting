import argparse
import socket
import time

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


def connect_with_retry(host, port, timeout_s=120.0, start_sleep_s=0.05, max_sleep_s=1.0):
    deadline = time.monotonic() + timeout_s
    sleep_s = start_sleep_s
    last_error = None
    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        configure_sock(sock, timeout_s=timeout_s)
        try:
            sock.connect((host, port))
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
            time.sleep(sleep_s)
            sleep_s = min(max_sleep_s, sleep_s * 2)
    raise TimeoutError(f"Timed out connecting to {host}:{port}; last_error={last_error}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)
    p.add_argument("--listen-port", type=int, required=True)
    p.add_argument("--next-host", required=True)
    p.add_argument("--next-port", type=int, required=True)
    p.add_argument("--max-nbytes", type=int, required=True)
    p.add_argument("--layer-start", type=int, required=True)
    p.add_argument("--layer-end", type=int, required=True)
    return p.parse_args()


def main():
    args = parse_args()

    if args.layer_end <= args.layer_start:
        raise ValueError(f"Invalid layer range: [{args.layer_start}, {args.layer_end})")

    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_sock.bind(("", args.listen_port))
    listen_sock.listen(1)
    configure_sock(listen_sock, timeout_s=120.0)

    import torch
    from transformers import AutoModelForCausalLM

    device = torch.device("cpu")
    dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype).to(device)
    model.eval()

    base = getattr(model, "model", model)
    layers_list = getattr(base, "layers", None)
    rotary_emb = getattr(base, "rotary_emb", None)
    if layers_list is None or rotary_emb is None:
        raise RuntimeError("Model does not expose expected layers/rotary_emb attributes.")

    num_blocks = len(layers_list)
    if args.layer_start < 0 or args.layer_end > num_blocks:
        raise ValueError(
            f"Layer range [{args.layer_start}, {args.layer_end}) out of bounds for {num_blocks} layers."
        )
    layers_to_run = layers_list[args.layer_start : args.layer_end]

    next_sock = connect_with_retry(args.next_host, args.next_port, timeout_s=120.0)
    upstream_sock, _ = listen_sock.accept()
    listen_sock.close()
    configure_sock(upstream_sock, timeout_s=120.0)
    configure_sock(next_sock, timeout_s=120.0)

    send_hello_expect_ready(
        next_sock,
        role="middle-worker",
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        max_nbytes=args.max_nbytes,
    )
    expect_hello_send_ready(upstream_sock, role="middle-worker", max_nbytes=args.max_nbytes)

    try:
        with torch.no_grad():
            while True:
                frame, tensor_bytes = recv_frame(upstream_sock, args.max_nbytes)
                kind = frame["kind"]
                if kind == KIND_SHUTDOWN:
                    send_frame(next_sock, {"kind": KIND_SHUTDOWN})
                    break
                if kind == KIND_ERROR:
                    send_frame(next_sock, frame)
                    break
                if kind != KIND_TENSOR:
                    raise RuntimeError(f"Unexpected frame kind from upstream: {frame}")

                shape = tuple(frame["shape"])
                hidden_states = (
                    torch.frombuffer(tensor_bytes, dtype=torch.float32)
                    .reshape(shape)
                    .clone()
                    .to(device)
                )

                seq_len = hidden_states.shape[1]
                position_ids = torch.arange(
                    seq_len, device=device, dtype=torch.long
                ).unsqueeze(0)
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

                for layer in layers_to_run:
                    hidden_states = layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_embeddings=position_embeddings,
                        position_ids=position_ids,
                        use_cache=False,
                    )

                out_nbytes = hidden_states.numel() * hidden_states.element_size()
                if out_nbytes > args.max_nbytes:
                    raise ValueError(
                        f"Output out_nbytes={out_nbytes} larger than max_nbytes={args.max_nbytes}"
                    )
                send_frame(
                    next_sock,
                    {
                        "kind": KIND_TENSOR,
                        "shape": tuple(hidden_states.shape),
                        "nbytes": out_nbytes,
                    },
                    hidden_states.cpu().numpy().tobytes(),
                )
    except Exception as exc:
        try:
            send_frame(
                next_sock,
                {"kind": KIND_ERROR, "message": f"{type(exc).__name__}: {exc}"},
            )
        except Exception:
            pass
        raise
    finally:
        upstream_sock.close()
        next_sock.close()


if __name__ == "__main__":
    main()
