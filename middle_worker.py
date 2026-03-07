import os
import sys
import pickle
import torch
from multiprocessing.shared_memory import SharedMemory
from transformers import AutoModelForCausalLM


def main():
    model_name = sys.argv[1]
    fd_read = int(sys.argv[2])
    fd_write = int(sys.argv[3])
    shm_in_name = sys.argv[4]
    shm_out_name = sys.argv[5]
    max_nbytes = int(sys.argv[6])
    k = int(sys.argv[7]) if len(sys.argv) > 7 else 0

    original_stdout = sys.stdout
    sys.stdout = sys.stderr
    device = torch.device("cpu")
    dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    sys.stdout = original_stdout
    model.eval()

    base = getattr(model, "model", model)
    layers_list = getattr(base, "layers", None)
    rotary_emb = getattr(base, "rotary_emb", None)
    assert layers_list is not None and rotary_emb is not None

    num_blocks = len(layers_list)
    layers_to_run = layers_list[k : num_blocks - k]

    shm_in = SharedMemory(name=shm_in_name, create=False)
    shm_out = SharedMemory(name=shm_out_name, create=False)
    pipe_from_parent = os.fdopen(fd_read, "rb")
    pipe_to_parent = os.fdopen(fd_write, "wb")

    with torch.no_grad():
        while True:
            meta = pickle.load(pipe_from_parent)
            if meta is None:
                break
            shape, nbytes = meta
            hidden_states = torch.frombuffer(
                memoryview(shm_in.buf)[:nbytes],
                dtype=torch.float32,
            ).reshape(shape).clone().to(device)
            seq_len = hidden_states.shape[1]
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
            position_embeddings = rotary_emb(hidden_states, position_ids)
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), torch.finfo(hidden_states.dtype).min, device=device, dtype=hidden_states.dtype),
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
            memoryview(shm_out.buf)[:out_nbytes][:] = hidden_states.cpu().numpy().tobytes()
            pickle.dump((hidden_states.shape, out_nbytes), pipe_to_parent)
            pipe_to_parent.flush()

    shm_in.close()
    shm_out.close()
    pipe_from_parent.close()
    pipe_to_parent.close()


if __name__ == "__main__":
    main()
