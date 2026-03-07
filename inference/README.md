# Inference

Distributed causal LM inference over two middle workers. The parent runs embed + first K and last K layers locally; the middle layers are split across W1 and W2, which are started independently and communicate via the frame protocol.

## Files

- `protocol.py`: frame protocol and HELLO/READY handshake
- `worker.py`: middle worker runtime
- `parent_client.py`: parent inference client (CLI)
- `Makefile`: commands to launch W1, W2, and run inference

## Prerequisites

- Active Python environment with `torch` and `transformers` installed (e.g. `conda activate quantenv`)
- Run all commands from `inference/`

## Run (three terminals)

From `inference/`:

Terminal 1:

```bash
make w2
```

Terminal 2:

```bash
make w1
```

Terminal 3:

```bash
make run
```

`make run` sends shutdown at the end, so W1 and W2 exit cleanly after one run.

## Optional overrides

```bash
make run PROMPT="What is 2+2?" NUM_NEW_TOKENS=8
```

If you change model or partitioning, keep worker ranges and `MAX_NBYTES` consistent with the Makefile (`W1_LAYER_START`, `W1_LAYER_END`, `W2_LAYER_START`, `W2_LAYER_END`, `MAX_NBYTES`).
