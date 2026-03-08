import pickle
import socket
import struct
import zlib

MAGIC = b"P5CH"
VERSION = 1
HEADER_STRUCT = struct.Struct(">4sBIII")
HEADER_SIZE = HEADER_STRUCT.size
KIND_HELLO = "hello"
KIND_READY = "ready"
KIND_TENSOR = "tensor"
KIND_SHUTDOWN = "shutdown"
KIND_ERROR = "error"


def configure_sock(sock, timeout_s=120.0):
    sock.settimeout(timeout_s)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)


def encode_frame(frame, tensor_bytes=None):
    payload = pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL)
    tensor_bytes = tensor_bytes or b""
    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    header = HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        len(payload),
        len(tensor_bytes),
        checksum,
    )
    return header + payload + tensor_bytes


def _validate_frame_blob(blob, max_nbytes):
    if len(blob) < HEADER_SIZE:
        raise ValueError(f"frame too short: {len(blob)} < {HEADER_SIZE}")

    magic, version, payload_len, tensor_len, checksum = HEADER_STRUCT.unpack(
        blob[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise ValueError(f"Bad frame magic: {magic!r}")
    if version != VERSION:
        raise ValueError(f"Unsupported frame version: {version}")
    if payload_len > 1_000_000:
        raise ValueError(f"payload too large: {payload_len}")
    if tensor_len > max_nbytes:
        raise ValueError(f"tensor_len={tensor_len} > max_nbytes={max_nbytes}")

    total_len = HEADER_SIZE + payload_len + tensor_len
    if len(blob) != total_len:
        raise ValueError(f"frame length mismatch: got {len(blob)}, expected {total_len}")

    payload_start = HEADER_SIZE
    payload_end = payload_start + payload_len
    payload = blob[payload_start:payload_end]
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

    tensor_bytes = blob[payload_end:] if tensor_len else None
    return frame, tensor_bytes


def decode_frame(blob, max_nbytes):
    return _validate_frame_blob(blob, max_nbytes)


def send_frame(sock, frame, tensor_bytes=None):
    sock.sendall(encode_frame(frame, tensor_bytes))


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
    header = recv_exact(sock, HEADER_SIZE)
    _, _, payload_len, tensor_len, _ = HEADER_STRUCT.unpack(header)
    payload = recv_exact(sock, payload_len)
    tensor_bytes = recv_exact(sock, tensor_len) if tensor_len else b""
    return decode_frame(header + payload + tensor_bytes, max_nbytes)


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
