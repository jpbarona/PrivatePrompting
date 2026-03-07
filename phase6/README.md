# Phase 6

Phase 6 keeps the same tensor framing/protocol and model partition math as Phase 5, but changes worker lifecycle so `W1` and `W2` are started independently (not spawned by the parent process).

## Files

- `protocol.py`: frame protocol and HELLO/READY handshake
- `worker.py`: middle worker runtime (same compute path as Phase 5)
- `parent_client.py`: parent inference client (CLI replacement for notebook driver)
- `Makefile`: commands to launch `W1`, `W2`, and run inference

## Prerequisites

- Active Python environment with `torch` and `transformers` installed (for conda, activate first)
- Run all commands from `phase6/`

## Run (three terminals)

From `phase6/`:

```bash
conda activate quantenv
```

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

`make run` sends shutdown at the end, so `w1` and `w2` exit cleanly after one run.

## Optional overrides

You can override defaults directly on the command line:

```bash
make run PROMPT="What is 2+2?" NUM_NEW_TOKENS=8
```

If you change model/partitioning, keep the worker ranges and `MAX_NBYTES` consistent:

- `W1_LAYER_START`, `W1_LAYER_END`
- `W2_LAYER_START`, `W2_LAYER_END`
- `MAX_NBYTES`
