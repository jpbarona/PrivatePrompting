import json
import time
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import hivemind
from hivemind.p2p import P2P
from hivemind.p2p.p2p_daemon_bindings.datastructures import PeerID
from multiaddr import Multiaddr
from hivemind.utils import get_dht_time

DEFAULT_DHT_TTL = 3600.0
DEFAULT_HANDLER_NAME = "inference_frame"


@dataclass
class WorkerInfo:
    role: str
    layer_start: int
    layer_end: int
    peer_id: str
    maddr: str
    handler_name: str = DEFAULT_HANDLER_NAME


def join_dht(
    *,
    dht_port: int,
    bootstrap_maddr: Optional[str],
    host_ip: str,
) -> Tuple[hivemind.DHT, str]:
    initial_peers = [bootstrap_maddr] if bootstrap_maddr else []
    dht = hivemind.DHT(
        host_maddrs=[f"/ip4/0.0.0.0/tcp/{dht_port}"],
        initial_peers=initial_peers,
        start=True,
    )
    visible = dht.get_visible_maddrs()
    if not visible:
        raise RuntimeError("No visible DHT addresses after startup")
    preferred = next((str(m) for m in visible if host_ip in str(m)), str(visible[0]))
    return dht, preferred


async def create_p2p_node(*, p2p_port: int) -> P2P:
    return await P2P.create(host_maddrs=[f"/ip4/0.0.0.0/tcp/{p2p_port}"])


def build_worker_info(
    *,
    role: str,
    host_ip: str,
    p2p_port: int,
    p2p: P2P,
    layer_start: int,
    layer_end: int,
    handler_name: str = DEFAULT_HANDLER_NAME,
) -> WorkerInfo:
    return WorkerInfo(
        role=role,
        layer_start=layer_start,
        layer_end=layer_end,
        peer_id=str(p2p.peer_id),
        maddr=f"/ip4/{host_ip}/tcp/{p2p_port}/p2p/{p2p.peer_id}",
        handler_name=handler_name,
    )


def register_worker(
    dht: hivemind.DHT,
    *,
    key: str,
    info: WorkerInfo,
    ttl_s: float = DEFAULT_DHT_TTL,
) -> None:
    dht.store(
        key,
        json.dumps(asdict(info)),
        expiration_time=get_dht_time() + ttl_s,
    )


def refresh_worker_registration(
    dht: hivemind.DHT,
    *,
    key: str,
    info: WorkerInfo,
    ttl_s: float = DEFAULT_DHT_TTL,
) -> None:
    register_worker(dht, key=key, info=info, ttl_s=ttl_s)


def discover_worker(
    dht: hivemind.DHT,
    *,
    key: str,
    retries: int = 30,
    sleep_s: float = 0.5,
) -> WorkerInfo:
    for _ in range(retries):
        result = dht.get(key, latest=True)
        if result is not None:
            payload = json.loads(result.value)
            return WorkerInfo(**payload)
        time.sleep(sleep_s)
    raise TimeoutError(f"Worker key '{key}' not found in DHT")


async def call_handler(
    *,
    p2p: P2P,
    peer_id: str,
    peer_maddr: Optional[str],
    handler_name: str,
    payload_bytes: bytes,
) -> bytes:
    remote_peer_id = PeerID.from_base58(peer_id)
    if peer_maddr:
        await p2p._client.connect(remote_peer_id, [Multiaddr(peer_maddr)])
    _, reader, writer = await p2p.call_binary_stream_handler(remote_peer_id, handler_name)
    try:
        await P2P.send_raw_data(payload_bytes, writer)
        return await P2P.receive_raw_data(reader)
    finally:
        writer.close()
