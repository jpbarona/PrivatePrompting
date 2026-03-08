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

HOST_IP ?= 127.0.0.1
BOOTSTRAP_MADDR ?= /ip4/127.0.0.1/tcp/43300/p2p/REPLACE_WITH_PEER0_DHT_PEER_ID

DHT_PORT_W1 ?= 43311
DHT_PORT_W2 ?= 43312
DHT_PORT_PARENT ?= 43313
P2P_PORT_W1 ?= 44211
P2P_PORT_W2 ?= 44212

DHT_KEY_W1 ?= inference_w1
DHT_KEY_W2 ?= inference_w2
HANDLER_NAME ?= inference_frame

.PHONY: help quickstart host_w2 host_w1 remote_run w1 w2 run

help:
	@echo "Run peers on host machine:"
	@echo "  1) make w2 BOOTSTRAP_MADDR=<peer0_dht_maddr> HOST_IP=<host_ip>"
	@echo "  2) make w1 BOOTSTRAP_MADDR=<peer0_dht_maddr> HOST_IP=<host_ip>"
	@echo "Run parent on remote machine:"
	@echo "  3) make run BOOTSTRAP_MADDR=<peer0_dht_maddr> HOST_IP=<client_ip>"
	@echo ""
	@echo "Convenience aliases:"
	@echo "  make host_w2      # same as w2"
	@echo "  make host_w1      # same as w1"
	@echo "  make remote_run   # same as run"
	@echo "  make quickstart   # prints copy/paste flow"

quickstart:
	@echo "Host terminal A:"
	@echo "  make host_w2 HOST_IP=<host_ip> BOOTSTRAP_MADDR=<peer0_dht_maddr>"
	@echo "Host terminal B:"
	@echo "  make host_w1 HOST_IP=<host_ip> BOOTSTRAP_MADDR=<peer0_dht_maddr>"
	@echo "Remote terminal:"
	@echo "  make remote_run HOST_IP=<client_ip> BOOTSTRAP_MADDR=<peer0_dht_maddr>"

w2:
	$(PYTHON) worker.py \
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
	$(PYTHON) worker.py \
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
	$(PYTHON) parent_client.py \
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
