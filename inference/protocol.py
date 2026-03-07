import pickle
import socket
import struct
import zlib

MAGIC = b"P5CH"
VERSION = 1
KIND_HELLO = "hello"
KIND_READY = "ready"
KIND_TENSOR = "tensor"
KIND_SHUTDOWN = "shutdown"
KIND_ERROR = "error"


def configure_sock(sock, timeout_s=120.0):
    sock.settimeout(timeout_s)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)


def send_frame(sock, frame, tensor_bytes=None):
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


def recv_exact(sock, n):
    buf = []
    while n > 0:
        chunk = sock.recv(n)
        if not chunk:
            raise ConnectionError("socket closed")
        buf.append(chunk)
        n -= len(chunk)
    return b"".join(buf)


def recv_frame(sock, max_nbytes):
    header = recv_exact(sock, 17)
    magic, version, payload_len, tensor_len, checksum = struct.unpack(">4sBIII", header)
    if magic != MAGIC:
        raise ValueError(f"Bad frame magic: {magic!r}")
    if version != VERSION:
        raise ValueError(f"Unsupported frame version: {version}")
    if payload_len > 1_000_000:
        raise ValueError(f"payload too large: {payload_len}")
    if tensor_len > max_nbytes:
        raise ValueError(f"tensor_len={tensor_len} > max_nbytes={max_nbytes}")

    payload = recv_exact(sock, payload_len)
    if (zlib.crc32(payload) & 0xFFFFFFFF) != checksum:
        raise ValueError("Frame checksum mismatch")

    frame = pickle.loads(payload)
    if not isinstance(frame, dict):
        raise ValueError(f"Frame must be a dict, got {type(frame).__name__}")
    if "kind" not in frame:
        raise ValueError("Frame missing 'kind'")
    kind = frame["kind"]
    if kind == KIND_TENSOR and "shape" not in frame:
        raise ValueError("tensor frame missing 'shape'")
    if kind == KIND_ERROR and "message" not in frame:
        raise ValueError("error frame missing 'message'")

    tensor_bytes = recv_exact(sock, tensor_len) if tensor_len else None
    return frame, tensor_bytes


def send_hello_expect_ready(sock, role, max_nbytes, layer_start=None, layer_end=None):
    frame = {"kind": KIND_HELLO, "role": role}
    if layer_start is not None:
        frame["layer_start"] = layer_start
    if layer_end is not None:
        frame["layer_end"] = layer_end
    send_frame(sock, frame)
    ready, _ = recv_frame(sock, max_nbytes)
    if ready["kind"] != KIND_READY:
        raise RuntimeError(f"Expected READY, got {ready}")


def expect_hello_send_ready(sock, role, max_nbytes):
    frame, _ = recv_frame(sock, max_nbytes)
    if frame["kind"] != KIND_HELLO:
        raise RuntimeError(f"Expected HELLO, got {frame}")
    send_frame(sock, {"kind": KIND_READY, "role": role})
