# Makefile for compiling CUDA kernels

# Get the current directory path
CURRENT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_DIR = $(abspath $(CURRENT_DIR)/../..)
BUILD_DIR = $(abspath $(CURRENT_DIR)/build)

$(info "Current directory: $(CURRENT_DIR)")
$(info "Project directory: $(PROJECT_DIR)")
$(info "Build directory: $(BUILD_DIR)")
all: torch

# kernel library
DEP_FILES = $(wildcard ${CURRENT_DIR}/csrc/library/*.cu)
DEP_FILES += $(wildcard ${CURRENT_DIR}/csrc/library/*.h)
${BUILD_DIR}/library/liblce.so:
	+$(MAKE) -f ${CURRENT_DIR}/csrc/library/Makefile
library: ${BUILD_DIR}/library/liblce.so

# torch wrapper
torch: library
	+$(MAKE) -f ${CURRENT_DIR}/csrc/torch/Makefile

clean:
	+$(MAKE) -f ${CURRENT_DIR}/csrc/library/Makefile clean
	+$(MAKE) -f ${CURRENT_DIR}/csrc/torch/Makefile clean
