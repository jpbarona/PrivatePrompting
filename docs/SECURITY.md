# Security Model

This document describes the security goals, trust assumptions, current protections, and realistic hardening options for this project.

## Threat Model

This project assumes:

- The client is trusted.
- The network is untrusted.
- Workers are untrusted and may be malicious.

The main privacy goal is to keep raw prompts and decoded outputs on the trusted client while using remote machines for middle-layer computation.

## What The Current Architecture Protects

With the current split-inference design, the following protections are realistic:

- Raw prompt text can stay on the client if tokenization and embedding remain local.
- Final decoded output text can stay on the client if final normalization, logits, and decoding remain local.
- Inference traffic sent over the Hivemind P2P path can be protected in transit by the underlying libp2p secure channel.
- The system can reduce direct exposure of user text to workers by sending hidden states instead of raw tokens.

In plain English: the current architecture is good at protecting data from passive network observers and at avoiding direct transmission of raw text to workers.

## What The Current Architecture Does Not Protect

The current architecture does not make workers "blind."

Specifically:

- A worker that computes on hidden states can see the hidden states it receives.
- Hidden states may still leak information about the original prompt or generation context.
- A malicious worker can return incorrect outputs, malformed outputs, or strategically manipulated outputs.
- Transport encryption does not hide IP addresses from the machines involved in the connection path.
- Discovery and metadata channels are only as private as the transport they use.

In plain English: current transport security can hide the messages on the wire, but it does not stop the worker that receives the data from inspecting it.

## What Exists Right Now

Today, the repository demonstrates a proof of concept for split inference with:

- A trusted client that can keep tokenization, embeddings, and decoding local.
- Remote workers that execute middle layers.
- P2P transport for inference messages between nodes.
- DHT-based worker discovery and registration.
- End-to-end orchestration for multi-process testing.

This is already useful for proving that split inference works and for reducing direct text exposure to remote machines.

## What Is Feasible With The Current Architecture

The following improvements are realistic without changing the basic split-inference architecture:

- Encrypt all discovery and metadata traffic, not just the inference stream.
- Route client traffic through a relay, VPN, or anonymity layer so workers do not directly learn the client IP.
- Add message authentication, request binding, nonces, and replay protection at the application layer.
- Add stricter framing limits, validation, and defensive parsing for all network messages.
- Add worker isolation, rate limiting, and timeout policies to reduce denial-of-service risk.
- Add redundancy or verification checks so the client can detect some classes of malicious worker behavior.
- Document the threat model and security guarantees clearly so the system does not over-promise privacy.

These are strong and practical hardening steps for the current design.

## Why Multiple Remote Runners Can Help

Using multiple independent remote runners can be more secure than using a single remote runner, even in the current architecture.

The main advantages are:

- No single remote runner has to execute the full remote segment.
- The client can compare outputs from different runners to detect some classes of malicious behavior.
- The client can send spot-check or duplicate work to test whether a worker is behaving honestly.
- A failure, outage, or refusal from one runner does not have to take down the whole remote path.

This does not make the system private against colluding workers, and it does not hide hidden states from the runner that processes them. However, it does reduce single-runner trust and creates opportunities for redundancy and lightweight verification.

## What Is Not Feasible Without A Major Redesign

The following goals are not realistically achievable with the current architecture alone:

- Preventing an arbitrary malicious worker from seeing the hidden states it processes.
- Guaranteeing that hidden states reveal nothing useful about the client input.
- Proving correctness of arbitrary worker computation without additional mechanisms.

Achieving those goals would require a different class of system, such as secure multi-party computation, homomorphic encryption, trusted execution environments, or specialized verification schemes.

## Practical Security Position

For this project, the practical and honest security position is:

- Trust the client.
- Do not trust the network.
- Do not trust workers.
- Keep raw text local whenever possible.
- Encrypt all traffic in transit.
- Minimize metadata exposure.
- Accept that workers processing hidden states may still learn something from those hidden states.

This is a meaningful privacy improvement over sending raw prompts and outputs to a remote inference service, but it is not the same as confidential computing.
