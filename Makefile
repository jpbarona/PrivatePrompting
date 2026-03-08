PYTHON ?= python

MODEL_NAME ?= Qwen/Qwen2.5-0.5B-Instruct
PROMPT ?= Hey! How are you feeling today?
NUM_NEW_TOKENS ?= 16
K ?= 2
NUM_MIDDLE_PARTITIONS ?= 2
MAX_NBYTES ?= 7340032
MAX_SEQ ?= 2048

W1_LAYER_START ?= 2
W1_LAYER_END ?= 12
W2_LAYER_START ?= 12
W2_LAYER_END ?= 22

HOST_IP ?= $(shell $(PYTHON) -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()")
BOOTSTRAP_MADDR ?= /ip4/127.0.0.1/tcp/43300/p2p/REPLACE_WITH_BOOTSTRAP_PEER_DHT_PEER_ID

DHT_PORT_W1 ?= 43311
DHT_PORT_W2 ?= 43312
DHT_PORT_PARENT ?= 43313
P2P_PORT_W1 ?= 44211
P2P_PORT_W2 ?= 44212
P2P_PORT_W1_E2E ?= 44221

RUN_ID ?=
DHT_KEY_BASE_W1 ?= inference_w1
DHT_KEY_BASE_W2 ?= inference_w2
DHT_KEY_W1 ?= $(DHT_KEY_BASE_W1)$(if $(RUN_ID),_$(RUN_ID),)
DHT_KEY_W2 ?= $(DHT_KEY_BASE_W2)$(if $(RUN_ID),_$(RUN_ID),)
HANDLER_NAME ?= inference_frame

.PHONY: help quickstart host_w2 host_w1 remote_run w1 w2 run test test-e2e test-integration bootstrap_peer

help:
	@echo "Run peers on host machine:"
	@echo "  0) make bootstrap_peer [HOST_IP=<lan_ip>]   # start DHT bootstrap (do this first)"
	@echo "  1) make w2 BOOTSTRAP_MADDR=<bootstrap_peer_dht_maddr> HOST_IP=<host_ip> [RUN_ID=<id>]"
	@echo "  2) make w1 BOOTSTRAP_MADDR=<bootstrap_peer_dht_maddr> HOST_IP=<host_ip> [RUN_ID=<id>]"
	@echo "Run parent on remote machine:"
	@echo "  3) make run BOOTSTRAP_MADDR=<bootstrap_peer_dht_maddr> HOST_IP=<client_ip> [RUN_ID=<id>]"
	@echo ""
	@echo "Convenience aliases:"
	@echo "  make host_w2      # same as w2"
	@echo "  make host_w1      # same as w1"
	@echo "  make remote_run   # same as run"
	@echo "  make test         # runs integration then e2e"
	@echo "  make test-e2e     # one-command orchestrated E2E"
	@echo "  make test-integration # frontend/backend integration contracts"
	@echo "  make quickstart   # prints copy/paste flow"
	@echo ""
	@echo "Safety notes:"
	@echo "  - If bootstrap_peer is running, avoid P2P_PORT_W1=44211 (conflicts with bootstrap peer P2P)."
	@echo "  - Set RUN_ID to isolate DHT keys per run (e.g. RUN_ID=v3)."
	@echo "  - Effective keys: DHT_KEY_W1=$(DHT_KEY_W1), DHT_KEY_W2=$(DHT_KEY_W2)"

quickstart:
	@echo "Set once (same value in all terminals): RUN_ID=<id> (example: RUN_ID=v3)"
	@echo "Host terminal A:"
	@echo "  make host_w2 HOST_IP=<host_ip> BOOTSTRAP_MADDR=<bootstrap_peer_dht_maddr> RUN_ID=<id>"
	@echo "Host terminal B:"
	@echo "  make host_w1 HOST_IP=<host_ip> BOOTSTRAP_MADDR=<bootstrap_peer_dht_maddr> RUN_ID=<id> P2P_PORT_W1=<non-conflicting_port>"
	@echo "Remote terminal:"
	@echo "  make remote_run HOST_IP=<client_ip> BOOTSTRAP_MADDR=<bootstrap_peer_dht_maddr> RUN_ID=<id>"

host_w2: w2

bootstrap_peer:
	$(PYTHON) p2p/bootstrap_peer.py --host-ip $(HOST_IP)

host_w1: w1

remote_run: run

w2:
	$(PYTHON) inference/worker.py \
		--role w2 \
		--model-name "$(MODEL_NAME)" \
		--host-ip $(HOST_IP) \
		--dht-port $(DHT_PORT_W2) \
		--p2p-port $(P2P_PORT_W2) \
		--bootstrap-maddr "$(BOOTSTRAP_MADDR)" \
		--dht-key "$(DHT_KEY_W2)" \
		--handler-name "$(HANDLER_NAME)" \
		--max-nbytes $(MAX_NBYTES) \
		--layer-start $(W2_LAYER_START) \
		--layer-end $(W2_LAYER_END)

w1:
	$(PYTHON) inference/worker.py \
		--role w1 \
		--model-name "$(MODEL_NAME)" \
		--host-ip $(HOST_IP) \
		--dht-port $(DHT_PORT_W1) \
		--p2p-port $(P2P_PORT_W1) \
		--bootstrap-maddr "$(BOOTSTRAP_MADDR)" \
		--dht-key "$(DHT_KEY_W1)" \
		--next-dht-key "$(DHT_KEY_W2)" \
		--handler-name "$(HANDLER_NAME)" \
		--max-nbytes $(MAX_NBYTES) \
		--layer-start $(W1_LAYER_START) \
		--layer-end $(W1_LAYER_END)

run:
	$(PYTHON) inference/parent_client.py \
		--model-name "$(MODEL_NAME)" \
		--prompt "$(PROMPT)" \
		--num-new-tokens $(NUM_NEW_TOKENS) \
		--k $(K) \
		--num-middle-partitions $(NUM_MIDDLE_PARTITIONS) \
		--host-ip $(HOST_IP) \
		--dht-port $(DHT_PORT_PARENT) \
		--bootstrap-maddr "$(BOOTSTRAP_MADDR)" \
		--w1-dht-key "$(DHT_KEY_W1)" \
		--w2-dht-key "$(DHT_KEY_W2)" \
		--max-seq $(MAX_SEQ)

test-e2e:
	@if [ "$(HOST_IP)" = "127.0.0.1" ] || [ -z "$(HOST_IP)" ]; then echo "HOST_IP must be your LAN IP (e.g. 192.168.x.x). Set HOST_IP= or ensure network is available for auto-detect."; exit 2; fi
	$(PYTHON) -m pytest tests/e2e -v \
		--host-ip "$(HOST_IP)" \
		--python "$(PYTHON)" \
		--model-name "$(MODEL_NAME)" \
		--prompt "$(PROMPT)" \
		--num-new-tokens $(NUM_NEW_TOKENS) \
		--k $(K) \
		--num-middle-partitions $(NUM_MIDDLE_PARTITIONS) \
		--max-nbytes $(MAX_NBYTES) \
		--max-seq $(MAX_SEQ) \
		--w1-layer-start $(W1_LAYER_START) \
		--w1-layer-end $(W1_LAYER_END) \
		--w2-layer-start $(W2_LAYER_START) \
		--w2-layer-end $(W2_LAYER_END) \
		--dht-port-w1 $(DHT_PORT_W1) \
		--dht-port-w2 $(DHT_PORT_W2) \
		--dht-port-parent $(DHT_PORT_PARENT) \
		--p2p-port-w1 $(P2P_PORT_W1_E2E) \
		--p2p-port-w2 $(P2P_PORT_W2) \
		--run-id "$(RUN_ID)"

test-integration:
	npm --prefix frontend test

test: test-integration test-e2e
