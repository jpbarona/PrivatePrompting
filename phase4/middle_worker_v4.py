import os
import pickle
import sys
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import resource_tracker

import torch
from transformers import AutoModelForCausalLM


def _safe_unregister_shm(shm_obj):
    try:
        resource_tracker.unregister(shm_obj._name, "shared_memory")
    except Exception:
        pass


def main():
    model_name = sys.argv[1]
    fd_left_read = int(sys.argv[2])
    fd_right_write = int(sys.argv[3])
    shm_left_name = sys.argv[4]
    shm_right_name = sys.argv[5]
    max_nbytes = int(sys.argv[6])
    layer_start = int(sys.argv[7])
    layer_end = int(sys.argv[8])

    if layer_end < layer_start:
        raise ValueError(f"Invalid layer range: [{layer_start}, {layer_end})")

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

    shm_left = SharedMemory(name=shm_left_name, create=False)
    shm_right = SharedMemory(name=shm_right_name, create=False)
    _safe_unregister_shm(shm_left)
    _safe_unregister_shm(shm_right)

    pipe_from_left = os.fdopen(fd_left_read, "rb")
    pipe_to_right = os.fdopen(fd_right_write, "wb")

    with torch.no_grad():
        while True:
            meta = pickle.load(pipe_from_left)
            if meta is None:
                pickle.dump(None, pipe_to_right)
                pipe_to_right.flush()
                break

            shape, nbytes = meta
            if nbytes > max_nbytes:
                raise ValueError(f"Received nbytes={nbytes} larger than max_nbytes={max_nbytes}")

            hidden_states = torch.frombuffer(
                memoryview(shm_left.buf)[:nbytes],
                dtype=torch.float32,
            ).reshape(shape).clone().to(device)

            seq_len = hidden_states.shape[1]
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
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
            memoryview(shm_right.buf)[:out_nbytes][:] = hidden_states.cpu().numpy().tobytes()
            pickle.dump((hidden_states.shape, out_nbytes), pipe_to_right)
            pipe_to_right.flush()

    shm_left.close()
    shm_right.close()
    pipe_from_left.close()
    pipe_to_right.close()


if __name__ == "__main__":
    main()
