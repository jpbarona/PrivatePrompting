# Inference

Distributed causal LM inference over two middle workers using raw Hivemind P2P transport. The parent runs embed + first K and last K layers locally; the middle layers are split across W1 and W2, discovered through DHT metadata and invoked over a binary stream handler while preserving the same frame protocol.

## Files

- `protocol.py`: frame protocol and HELLO/READY handshake
- `p2p_transport.py`: DHT/P2P adapter for registration, discovery, and calls
- `worker.py`: middle worker runtime (P2P handler)
- `parent_client.py`: parent inference client (CLI over P2P)
- `Makefile`: commands to launch W1, W2, and run inference

## Prerequisites

- From repo root, run `make setup` to create `.venv` (Python 3.11) and install pip dependencies
- Run all `make` commands from repo root (`Hackathon/`)
- A bootstrap DHT multiaddr is required (for MVP this is the bootstrap peer's DHT maddr from `p2p/bootstrap_peer.py`)

## Run E2E (host peers + remote client)

### Fast path (recommended)

Use the Makefile convenience aliases:

```bash
make quickstart
```

Then execute exactly what it prints:
- Host terminal A: `make host_w2 HOST_IP=<host_ip> BOOTSTRAP_MADDR=<bootstrap_peer_dht_maddr>`
- Host terminal B: `make host_w1 HOST_IP=<host_ip> BOOTSTRAP_MADDR=<bootstrap_peer_dht_maddr>`
- Remote terminal: `make remote_run HOST_IP=<client_ip> BOOTSTRAP_MADDR=<bootstrap_peer_dht_maddr>`

### 1) On host machine (runs W2 then W1)

First export the host machine LAN IP and bootstrap maddr:

```bash
export HOST_IP=192.168.1.31
export BOOTSTRAP_MADDR="/ip4/192.168.1.31/tcp/43300/p2p/<bootstrap_peer_dht_peer_id>"
```

Then start workers:

```bash
make w2
```

```bash
make w1
```

Worker logs should show:
- DHT address and advertised P2P maddr
- DHT key (`inference_w1` / `inference_w2`)
- layer range and handler name (`inference_frame`)

### 2) On remote machine (runs parent client)

Export client host IP and same bootstrap maddr:

```bash
export HOST_IP=<remote_client_ip>
export BOOTSTRAP_MADDR="/ip4/192.168.1.31/tcp/43300/p2p/<bootstrap_peer_dht_peer_id>"
```

Run:

```bash
make run
```

`make run` discovers W1/W2 from DHT, sends HELLO->READY, performs inference, prints baseline vs split parity, then sends SHUTDOWN (workers exit cleanly).

## Optional overrides

```bash
make run PROMPT="What is 2+2?" NUM_NEW_TOKENS=8
```

If you change model or partitioning, keep worker ranges and sizing consistent (`W1_LAYER_START`, `W1_LAYER_END`, `W2_LAYER_START`, `W2_LAYER_END`, `MAX_NBYTES`, `MAX_SEQ`).

## Network requirements

- Host machine must accept inbound TCP for:
  - DHT: `DHT_PORT_W1`, `DHT_PORT_W2`
  - P2P handlers: `P2P_PORT_W1`, `P2P_PORT_W2`
- Both machines must reach the bootstrap DHT multiaddr.
- Use LAN-reachable `HOST_IP` values (avoid loopback for cross-machine runs).