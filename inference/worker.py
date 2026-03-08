import argparse
import asyncio

from hivemind.p2p import P2P

from protocol import (
    KIND_ERROR,
    KIND_HELLO,
    KIND_READY,
    KIND_SHUTDOWN,
    KIND_TENSOR,
    decode_frame,
    encode_frame,
)
from p2p_transport import (
    DEFAULT_DHT_TTL,
    DEFAULT_HANDLER_NAME,
    build_worker_info,
    call_handler,
    create_p2p_node,
    discover_worker,
    join_dht,
    refresh_worker_registration,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=("w1", "w2"), required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--host-ip", required=True)
    p.add_argument("--dht-port", type=int, required=True)
    p.add_argument("--p2p-port", type=int, required=True)
    p.add_argument("--bootstrap-maddr", default=None)
    p.add_argument("--dht-key", required=True)
    p.add_argument("--next-dht-key", default=None)
    p.add_argument("--handler-name", default=DEFAULT_HANDLER_NAME)
    p.add_argument("--max-nbytes", type=int, required=True)
    p.add_argument("--layer-start", type=int, required=True)
    p.add_argument("--layer-end", type=int, required=True)
    p.add_argument("--dht-ttl", type=float, default=DEFAULT_DHT_TTL)
    return p.parse_args()


async def main():
    args = parse_args()
    if args.layer_end <= args.layer_start:
        raise ValueError(f"Invalid layer range: [{args.layer_start}, {args.layer_end})")
    if args.role == "w1" and not args.next_dht_key:
        raise ValueError("--next-dht-key is required for role=w1")

    import torch
    from transformers import AutoModelForCausalLM

    device = torch.device("cpu")
    dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype).to(device)
    model.eval()

    base = getattr(model, "model", model)
    layers_list = getattr(base, "layers", None)
    rotary_emb = getattr(base, "rotary_emb", None)
    if layers_list is None or rotary_emb is None:
        raise RuntimeError("Model does not expose expected layers/rotary_emb attributes.")

    num_blocks = len(layers_list)
    if args.layer_start < 0 or args.layer_end > num_blocks:
        raise ValueError(
            f"Layer range [{args.layer_start}, {args.layer_end}) out of bounds for {num_blocks} layers."
        )
    layers_to_run = layers_list[args.layer_start : args.layer_end]

    dht, dht_maddr = join_dht(
        dht_port=args.dht_port,
        bootstrap_maddr=args.bootstrap_maddr,
        host_ip=args.host_ip,
    )
    p2p = await create_p2p_node(p2p_port=args.p2p_port)
    info = build_worker_info(
        role=args.role,
        host_ip=args.host_ip,
        p2p_port=args.p2p_port,
        p2p=p2p,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        handler_name=args.handler_name,
    )
    refresh_worker_registration(dht, key=args.dht_key, info=info, ttl_s=args.dht_ttl)

    downstream = None
    downstream_ready = False
    if args.next_dht_key:
        downstream = discover_worker(dht, key=args.next_dht_key)
        if downstream.layer_start != args.layer_end:
            raise RuntimeError(
                f"Layer continuity mismatch: current end={args.layer_end}, next start={downstream.layer_start}"
            )

    stop_event = asyncio.Event()

    async def run_local_layers(frame, tensor_bytes):
        shape = tuple(frame["shape"])
        hidden_states = (
            torch.frombuffer(tensor_bytes, dtype=torch.float32).reshape(shape).clone().to(device)
        )
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
        if out_nbytes > args.max_nbytes:
            raise ValueError(
                f"Output out_nbytes={out_nbytes} larger than max_nbytes={args.max_nbytes}"
            )
        return {
            "kind": KIND_TENSOR,
            "shape": tuple(hidden_states.shape),
            "nbytes": out_nbytes,
        }, hidden_states.cpu().numpy().tobytes()

    async def handle(stream_info, reader, writer):
        nonlocal downstream_ready
        try:
            inbound = await P2P.receive_raw_data(reader)
            frame, tensor_bytes = decode_frame(inbound, args.max_nbytes)
            kind = frame["kind"]

            if kind == KIND_HELLO:
                if downstream and not downstream_ready:
                    hello_blob = encode_frame(
                        {
                            "kind": KIND_HELLO,
                            "role": args.role,
                            "layer_start": args.layer_start,
                            "layer_end": args.layer_end,
                        }
                    )
                    downstream_reply = await call_handler(
                        p2p=p2p,
                        peer_id=downstream.peer_id,
                        handler_name=downstream.handler_name,
                        payload_bytes=hello_blob,
                    )
                    downstream_frame, _ = decode_frame(downstream_reply, args.max_nbytes)
                    if downstream_frame["kind"] != KIND_READY:
                        raise RuntimeError(f"Expected READY from downstream, got {downstream_frame}")
                    downstream_ready = True
                response_blob = encode_frame({"kind": KIND_READY, "role": args.role})
                await P2P.send_raw_data(response_blob, writer)
                return

            if kind == KIND_SHUTDOWN:
                if downstream:
                    await call_handler(
                        p2p=p2p,
                        peer_id=downstream.peer_id,
                        handler_name=downstream.handler_name,
                        payload_bytes=encode_frame({"kind": KIND_SHUTDOWN}),
                    )
                await P2P.send_raw_data(encode_frame({"kind": KIND_SHUTDOWN}), writer)
                stop_event.set()
                return

            if kind == KIND_ERROR:
                if downstream:
                    await call_handler(
                        p2p=p2p,
                        peer_id=downstream.peer_id,
                        handler_name=downstream.handler_name,
                        payload_bytes=encode_frame(frame),
                    )
                await P2P.send_raw_data(encode_frame(frame), writer)
                stop_event.set()
                return

            if kind != KIND_TENSOR:
                raise RuntimeError(f"Unexpected frame kind: {frame}")

            local_frame, local_tensor = await run_local_layers(frame, tensor_bytes)
            if downstream:
                downstream_reply = await call_handler(
                    p2p=p2p,
                    peer_id=downstream.peer_id,
                    handler_name=downstream.handler_name,
                    payload_bytes=encode_frame(local_frame, local_tensor),
                )
                await P2P.send_raw_data(downstream_reply, writer)
                return

            await P2P.send_raw_data(encode_frame(local_frame, local_tensor), writer)
        except Exception as exc:
            error_blob = encode_frame({"kind": KIND_ERROR, "message": f"{type(exc).__name__}: {exc}"})
            try:
                await P2P.send_raw_data(error_blob, writer)
            finally:
                stop_event.set()
                raise
        finally:
            writer.close()

    await p2p.add_binary_stream_handler(args.handler_name, handle)
    print(
        f"[{args.role}] DHT={dht_maddr} P2P={info.maddr} key={args.dht_key} "
        f"layers=[{args.layer_start},{args.layer_end}) handler={args.handler_name}"
    )

    async def refresh_loop():
        while not stop_event.is_set():
            await asyncio.sleep(args.dht_ttl / 2)
            refresh_worker_registration(dht, key=args.dht_key, info=info, ttl_s=args.dht_ttl)

    refresh_task = asyncio.create_task(refresh_loop())
    try:
        await stop_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        refresh_task.cancel()
        await p2p.shutdown()
        dht.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
