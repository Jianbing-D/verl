MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_PATH = $(abspath ${MAKEFILE_DIR}/../)
BUILD_DIR := $(abspath ${PROJECT_PATH}/../build/library)

NVCC ?= $(shell which nvcc)
SM ?= 80 90a
CXX_STANDARD ?= 17

all: build

# --------- # submodule cannot specify tag, so we need to clone the repo and checkout the tag manually
CUTLASS_REPO := https://github.com/NVIDIA/cutlass.git
CUTLASS_TAG := v3.7.0
CUTLASS_PATH := $(abspath ${PROJECT_PATH}/3rd_party/cutlass)
get_cutlass:
	@if [ ! -d "${CUTLASS_PATH}" ]; then \
		git clone -b ${CUTLASS_TAG} ${CUTLASS_REPO} ${CUTLASS_PATH}; \
	fi

# ------- args --------- #
SmOption = $(foreach sm,$(patsubst "","",${SM}),-gencode arch=compute_${sm},code=sm_${sm})
NVCC_FLAGS := -std=c++${CXX_STANDARD}
NVCC_FLAGS += -O3 -lineinfo
NVCC_FLAGS += ${SmOption}
NVCC_FLAGS += -D_GLIBCXX_USE_CXX11_ABI=1
NVCC_FLAGS += --Werror all-warnings
NVCC_FLAGS += --extended-lambda
NVCC_FLAGS += --expt-relaxed-constexpr
NVCC_FLAGS += -Xcompiler -fPIC

INC_PATH := $(abspath ${CUTLASS_PATH}/include)
INC_PATH += $(abspath ${MAKEFILE_DIR}/)
_REAL_INC_PATH = $(foreach path,$(patsubst "","",${INC_PATH}),-I${path})

DEP_FILES = $(wildcard ${MAKEFILE_DIR}/*.cu)
DEP_FILES += $(wildcard ${MAKEFILE_DIR}/*.h)


SRC_FILES = $(wildcard ${MAKEFILE_DIR}/*.cu)

SRC_FILES_NO_PREFIX = $(patsubst ${MAKEFILE_DIR}/%,%,${SRC_FILES})
SRC_FILES_NO_PREFIX_NO_SUFFIX = $(basename ${SRC_FILES_NO_PREFIX})
SRC_OBJS = $(addsuffix .obj,${SRC_FILES_NO_PREFIX_NO_SUFFIX})
SRC_OBJS_FULL = $(addprefix ${BUILD_DIR}/,${SRC_OBJS})

SRC_LIB = ${BUILD_DIR}/liblce.so

${BUILD_DIR}/%.obj: ${MAKEFILE_DIR}/%.cu ${DEP_FILES}
	mkdir -p $(abspath $(dir $@))
	${NVCC} ${NVCC_FLAGS} -c $< -o $@ ${_REAL_INC_PATH}
${SRC_LIB}: ${SRC_OBJS_FULL} ${SRC_FILES}
	${NVCC} -shared -o $@ ${SRC_OBJS_FULL}

build: get_cutlass ${SRC_LIB}

clean:
	@rm -rf ${BUILD_DIR}
	# @rm -rf ${CUTLASS_PATH}





