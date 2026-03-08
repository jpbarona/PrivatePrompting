# Hackathon

Root commands for P2P inference live in `Makefile`.

## Quickstart

From repo root:

```bash
make setup
make quickstart
```

Or run directly in 3 terminals:

```bash
# Host terminal A (worker 2)
make host_w2 HOST_IP=<host_ip> BOOTSTRAP_MADDR=<bootstrap_peer_dht_maddr>

# Host terminal B (worker 1)
make host_w1 HOST_IP=<host_ip> BOOTSTRAP_MADDR=<bootstrap_peer_dht_maddr>

# Remote terminal (parent/client)
make remote_run HOST_IP=<client_ip> BOOTSTRAP_MADDR=<bootstrap_peer_dht_maddr>
```

## Required parameters

- `HOST_IP`
  - Host workers: LAN IP of the machine running peers/workers (example: `192.168.1.31`).
  - Remote client: LAN IP of the remote machine running `make remote_run`.
- `BOOTSTRAP_MADDR`
  - Full DHT bootstrap multiaddress in the format:
    - `/ip4/<host_ip>/tcp/<dht_port>/p2p/<peer_id>`
  - For your setup, this is the bootstrap peer's DHT maddr.

## How to find `HOST_IP`

- macOS:
  - `ipconfig getifaddr en0` (Wi-Fi) or `ipconfig getifaddr en1` (alternate interface)
- Use the address reachable by your friend over LAN.

## How to find `BOOTSTRAP_MADDR`

Use one of these:

1) From bootstrap peer startup logs (preferred):
- Look for `DHT bootstrap maddr: /ip4/.../tcp/.../p2p/...`

2) From bootstrap peer HTTP endpoint:
- If the bootstrap peer is running (`p2p/bootstrap_peer.py`), fetch:
  - `http://<host_ip>:8765/`
- Example:
  - `curl "http://192.168.1.31:8765/"`

## Useful commands

- `make setup` - create `.venv`, install dependencies, and set local runtime baseline.
- `make help` - print all command usage.
- `make w2 ...` / `make w1 ...` / `make run ...` - base targets.
- `make host_w2 ...` / `make host_w1 ...` / `make remote_run ...` - convenience aliases.

## Notes

- Keep `BOOTSTRAP_MADDR` identical across all three terminals.
- Start `w2` before `w1`.
- Open firewall/port access for DHT and P2P worker ports if cross-machine connectivity fails.
- Inference-specific details remain in `inference/README.md`.
