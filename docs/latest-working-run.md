# Latest Working Distributed Inference Run

Date: 2026-03-08

## Environment

- Host machine IP: `192.168.1.31`
- Python env: `quantenv`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Topology:
  - `bootstrap_peer` as DHT bootstrap
  - `w2` middle partition `[12,22)`
  - `w1` middle partition `[2,12)`
  - `parent_client` runs first/last `k=2` layers

## Bootstrap Values Used

- DHT bootstrap maddr from bootstrap peer log:
  - `/ip4/192.168.1.31/tcp/43300/p2p/12D3KooWSp92987oeFmMYFBQczt1x6ARcb1qd6W22Gwt1TNK9KED`

## Commands Used

Terminal 1 (keep running):

```bash
conda activate quantenv
python p2p/bootstrap_peer.py --host-ip 192.168.1.31
```

Terminal 2:

```bash
conda activate quantenv
make host_w2 HOST_IP=192.168.1.31 BOOTSTRAP_MADDR="/ip4/192.168.1.31/tcp/43300/p2p/12D3KooWSp92987oeFmMYFBQczt1x6ARcb1qd6W22Gwt1TNK9KED" DHT_KEY_W2=inference_w2_v2
```

Terminal 3:

```bash
conda activate quantenv
make host_w1 HOST_IP=192.168.1.31 BOOTSTRAP_MADDR="/ip4/192.168.1.31/tcp/43300/p2p/12D3KooWSp92987oeFmMYFBQczt1x6ARcb1qd6W22Gwt1TNK9KED" DHT_KEY_W1=inference_w1_v2 DHT_KEY_W2=inference_w2_v2 P2P_PORT_W1=44221
```

Terminal 4:

```bash
conda activate quantenv
make remote_run HOST_IP=192.168.1.31 BOOTSTRAP_MADDR="/ip4/192.168.1.31/tcp/43300/p2p/12D3KooWSp92987oeFmMYFBQczt1x6ARcb1qd6W22Gwt1TNK9KED" DHT_KEY_W1=inference_w1_v2 DHT_KEY_W2=inference_w2_v2 PROMPT="What is the capital of France?"
```

## Why This Run Worked

- Used bootstrap DHT maddr (`tcp/43300`), not bootstrap peer P2P maddr (`tcp/44211`).
- Avoided stale key collisions by using fresh DHT keys (`inference_w1_v2`, `inference_w2_v2`).
- Avoided bootstrap peer/`w1` P2P port conflict by setting `P2P_PORT_W1=44221`.
- Parent and workers were updated to dial peers via discovered maddr before opening handler streams.
- Worker forward path now serializes detached/no-grad tensors.

## Success Signal

The run is considered successful when parent prints:

- `match: True`
- Baseline token IDs equal split token IDs
- Baseline decoded text equals split decoded text
- Workers remain running after `remote_run` and are stopped manually by each volunteer.
