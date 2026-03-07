PYTHON ?= python

MODEL_NAME ?= Qwen/Qwen2.5-0.5B-Instruct
PROMPT ?= Hey! How are you feeling today?
NUM_NEW_TOKENS ?= 16
K ?= 2
NUM_MIDDLE_PARTITIONS ?= 2
PORT_PARENT ?= 5100
PORT_W1 ?= 5101
PORT_W2 ?= 5102
MAX_NBYTES ?= 7340032

W1_LAYER_START ?= 2
W1_LAYER_END ?= 12
W2_LAYER_START ?= 12
W2_LAYER_END ?= 22

.PHONY: help w1 w2 run

help:
	@echo "Run in 3 terminals:"
	@echo "  1) make w2"
	@echo "  2) make w1"
	@echo "  3) make run"

w2:
	$(PYTHON) worker.py \
		--model-name "$(MODEL_NAME)" \
		--listen-port $(PORT_W2) \
		--next-host 127.0.0.1 \
		--next-port $(PORT_PARENT) \
		--max-nbytes $(MAX_NBYTES) \
		--layer-start $(W2_LAYER_START) \
		--layer-end $(W2_LAYER_END)

w1:
	$(PYTHON) worker.py \
		--model-name "$(MODEL_NAME)" \
		--listen-port $(PORT_W1) \
		--next-host 127.0.0.1 \
		--next-port $(PORT_W2) \
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
		--port-parent $(PORT_PARENT) \
		--port-w1 $(PORT_W1)
