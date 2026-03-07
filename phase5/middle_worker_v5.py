import pickle
import socket
import struct
import sys
import time
import zlib

MAGIC = b"P5CH"
VERSION = 1
KIND_HELLO = "hello"
KIND_READY = "ready"
KIND_TENSOR = "tensor"
KIND_SHUTDOWN = "shutdown"
KIND_ERROR = "error"


def _send_frame(sock, frame, tensor_bytes=None):
    payload = pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL)
    tensor_bytes = tensor_bytes or b""
    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    header = struct.pack(
        ">4sBIII",
        MAGIC,
        VERSION,
        len(payload),
        len(tensor_bytes),
        checksum,
    )
    sock.sendall(header)
    sock.sendall(payload)
    if tensor_bytes:
        sock.sendall(tensor_bytes)


def _recv_frame(sock, max_nbytes):
    header = _recv_exact(sock, 17)
    magic, version, payload_len, tensor_len, checksum = struct.unpack(
        ">4sBIII", header
    )
    if magic != MAGIC:
        raise ValueError(f"Bad frame magic: {magic!r}")
    if version != VERSION:
        raise ValueError(f"Unsupported frame version: {version}")
    if payload_len > 1_000_000:
        raise ValueError(f"Payload too large: {payload_len}")
    if tensor_len > max_nbytes:
        raise ValueError(f"tensor_len={tensor_len} > max_nbytes={max_nbytes}")

    payload = _recv_exact(sock, payload_len)
    actual_checksum = zlib.crc32(payload) & 0xFFFFFFFF
    if actual_checksum != checksum:
        raise ValueError("Frame checksum mismatch")

    frame = pickle.loads(payload)
    tensor_bytes = _recv_exact(sock, tensor_len) if tensor_len else None
    return frame, tensor_bytes


def _recv_exact(sock, n):
    buf = []
    while n > 0:
        chunk = sock.recv(n)
        if not chunk:
            raise ConnectionError("socket closed")
        buf.append(chunk)
        n -= len(chunk)
    return b"".join(buf)


def _configure_sock(sock, timeout_s):
    sock.settimeout(timeout_s)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)


def _connect_with_retry(host, port, timeout_s=120.0, start_sleep_s=0.05, max_sleep_s=1.0):
    deadline = time.monotonic() + timeout_s
    sleep_s = start_sleep_s
    last_error = None
    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _configure_sock(sock, timeout_s)
        try:
            sock.connect((host, port))
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
            time.sleep(sleep_s)
            sleep_s = min(max_sleep_s, sleep_s * 2)
    raise TimeoutError(
        f"Timed out connecting to {host}:{port}; last_error={last_error}"
    )


def _send_hello_expect_ready(sock, role, layer_start, layer_end, max_nbytes):
    _send_frame(
        sock,
        {
            "kind": KIND_HELLO,
            "role": role,
            "layer_start": layer_start,
            "layer_end": layer_end,
        },
    )
    frame, _ = _recv_frame(sock, max_nbytes)
    if frame.get("kind") != KIND_READY:
        raise RuntimeError(f"Expected READY, got {frame}")


def _expect_hello_send_ready(sock, role, max_nbytes):
    frame, _ = _recv_frame(sock, max_nbytes)
    if frame.get("kind") != KIND_HELLO:
        raise RuntimeError(f"Expected HELLO, got {frame}")
    _send_frame(sock, {"kind": KIND_READY, "role": role})


def main():
    model_name = sys.argv[1]
    listen_port = int(sys.argv[2])
    next_host = sys.argv[3]
    next_port = int(sys.argv[4])
    max_nbytes = int(sys.argv[5])
    layer_start = int(sys.argv[6])
    layer_end = int(sys.argv[7])

    if layer_end <= layer_start:
        raise ValueError(f"Invalid layer range: [{layer_start}, {layer_end})")

    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_sock.bind(("", listen_port))
    listen_sock.listen(1)
    _configure_sock(listen_sock, 120.0)

    import torch
    from transformers import AutoModelForCausalLM

    original_stdout = sys.stdout
    sys.stdout = sys.stderr
    device = torch.device("cpu")
    dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    sys.stdout = original_stdout
    model.eval()

    base = getattr(model, "model", model)
    layers_list = getattr(base, "layers", None)
    rotary_emb = getattr(base, "rotary_emb", None)
    if layers_list is None or rotary_emb is None:
        raise RuntimeError("Model does not expose expected layers/rotary_emb attributes.")

    num_blocks = len(layers_list)
    if layer_start < 0 or layer_end > num_blocks:
        raise ValueError(
            f"Layer range [{layer_start}, {layer_end}) out of bounds for {num_blocks} layers."
        )
    layers_to_run = layers_list[layer_start:layer_end]

    next_sock = _connect_with_retry(next_host, next_port, timeout_s=120.0)

    upstream_sock, _ = listen_sock.accept()
    listen_sock.close()
    _configure_sock(upstream_sock, 120.0)
    _configure_sock(next_sock, 120.0)

    _send_hello_expect_ready(
        next_sock,
        role="middle-worker",
        layer_start=layer_start,
        layer_end=layer_end,
        max_nbytes=max_nbytes,
    )
    _expect_hello_send_ready(upstream_sock, role="middle-worker", max_nbytes=max_nbytes)

    try:
        with torch.no_grad():
            while True:
                frame, tensor_bytes = _recv_frame(upstream_sock, max_nbytes)
                kind = frame.get("kind")
                if kind == KIND_SHUTDOWN:
                    _send_frame(next_sock, {"kind": KIND_SHUTDOWN})
                    break
                if kind == KIND_ERROR:
                    _send_frame(next_sock, frame)
                    break
                if kind != KIND_TENSOR:
                    raise RuntimeError(f"Unexpected frame kind from upstream: {frame}")

                shape = tuple(frame["shape"])
                nbytes = int(frame["nbytes"])
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
                if out_nbytes > max_nbytes:
                    raise ValueError(
                        f"Output out_nbytes={out_nbytes} larger than max_nbytes={max_nbytes}"
                    )
                _send_frame(
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
            _send_frame(
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
