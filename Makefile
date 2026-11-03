# torch.cr build & test
#
#   make ext      build the C shim (ext/libtch.so, STATIC=1 for libtch.a)
#   make fixture  regenerate the TorchScript fixtures with python torch
#   make spec     run specs (builds ext + fixture if needed)
#   make oracle   build + dump current tree (oracle/current)
#   make verify   twinshot review: regenerate both trees, compare
#   make clean

TORCH_DIR ?= $(shell ./bin/detect_torch.sh 2>/dev/null)
PY ?= $(shell if [ -x "$(HOME)/dataset-vox-cpm2/.venv/bin/python" ]; then echo "$(HOME)/dataset-vox-cpm2/.venv/bin/python"; else echo python3; fi)

TORCH_LIB := $(TORCH_DIR)/lib
CUDA_LIBS := $(if $(wildcard $(TORCH_LIB)/libtorch_cuda.so),-ltorch_cuda -lc10_cuda,)
EXT_DIR := $(abspath ext)

# Flags for linking a binary. With the default ext/libtch.so the shim is
# self-contained (rpath into torch/lib); with STATIC=1 ext/build.sh, use
# LINK_FLAGS_STATIC instead.
LINK_FLAGS := -L$(EXT_DIR) -Wl,-rpath,$(EXT_DIR) -lstdc++
LINK_FLAGS_STATIC := -L$(EXT_DIR) -L$(TORCH_LIB) -Wl,-rpath,$(TORCH_LIB) -ltorch -ltorch_cpu -lc10 $(CUDA_LIBS) -lstdc++

.PHONY: ext fixture spec oracle verify clean print-link-flags

ext:
	./ext/build.sh

fixture:
	$(PY) spec/support/gen_fixture.py

print-link-flags:
	@echo "-L$(EXT_DIR) -Wl,-rpath,$(EXT_DIR) -lstdc++"

oracle: ext fixture
	crystal build --release -o bin/oracle_dump --link-flags "$(LINK_FLAGS)" bin/oracle_dump.cr
	./bin/oracle_dump

TWINSHOT := $(shell command -v twinshot 2>/dev/null || echo "$(HOME)/.gem/ruby/3.3.0/bin/twinshot")

verify: oracle fixture
	$(TWINSHOT) review --no-open

spec: ext fixture
	crystal spec --link-flags "$(LINK_FLAGS)"

clean:
	rm -f ext/libtch.a ext/libtch.so bin/oracle_dump
	rm -rf oracle/current
