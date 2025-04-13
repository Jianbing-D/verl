MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BUILD_DIR := $(abspath ${MAKEFILE_DIR}/../../build/torch)

$(info "torch build directory: $(BUILD_DIR)")

all: build

DEP_FILES = $(wildcard ${MAKEFILE_DIR}/*.cu)
build: ${BUILD_DIR}/build.stamp ${DEP_FILES}

INSTALL_LOG = ${BUILD_DIR}/torch_install.log
${BUILD_DIR}/build.stamp: ${DEP_FILES}
	mkdir -p $(abspath $(dir ${INSTALL_LOG}))
	pip install -v -e ${MAKEFILE_DIR} 2>&1 | tee ${INSTALL_LOG}
	touch $@

clean:
	@rm -rf ${BUILD_DIR}/build.stamp
	@rm -rf ${BUILD_DIR}/torch_install.log
	@rm -rf ${MAKEFILE_DIR}/build/
	@rm -rf ${MAKEFILE_DIR}/*.egg-info
	@rm -rf ${MAKEFILE_DIR}/*.so

