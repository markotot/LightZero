GIT_BRANCH = main
PROJECT_NAME = LightZero

#APOCRITA
AP_PRIVATE_KEY_PATH = ~/Apocrita/apocrita.ssh
APOCRITA_USER = acw549

#EXPERIMENT CONFIG
START_SEED = 1
END_SEED = 10
RUN_NAME = "iris"

# Used to login to apocrita server
.SILENT: apocrita_login
apocrita_login:
	sudo expect ./scripts/apocrita_login.sh \
	${APOCRITA_USER} ${APOCRITA_PASSPHRASE} ${APOCRITA_USER_PASSWORD} ${AP_PRIVATE_KEY_PATH}

# Clones the repository from github to apocrita (use only once)
.SILENT: apocrita_clone_repo
apocrita_clone_repo:
	sudo expect ./scripts/apocrita_clone_repo.sh \
	${APOCRITA_USER} ${APOCRITA_PASSPHRASE} ${APOCRITA_USER_PASSWORD} ${AP_PRIVATE_KEY_PATH} \
 	${GIT_BRANCH} ${GITHUB_USER} ${GITHUB_TOKEN} ${PROJECT_NAME}

# Aggregates the results of the main.py on apocrita using apptainer
.SILENT: apocrita_aggregate
apocrita_aggregate:
	sudo expect ./scripts/apocrita_aggregate.sh \
 	${APOCRITA_USER} ${APOCRITA_PASSPHRASE} ${APOCRITA_USER_PASSWORD} ${AP_PRIVATE_KEY_PATH} \
 	${GIT_BRANCH} ${PROJECT_NAME} ${WANDB_API_TOKEN} ${START_SEED} ${END_SEED} ${RUN_NAME}

.SILENT: apocrita_build
apocrita_build:
	sudo expect ./scripts/apocrita_build.sh \
 	${APOCRITA_USER} ${APOCRITA_PASSPHRASE} ${APOCRITA_USER_PASSWORD} ${AP_PRIVATE_KEY_PATH} \
 	${GIT_BRANCH} ${PROJECT_NAME} ${WANDB_API_TOKEN}

# Builds and runs the main.py on apocrita using apptainer
.SILENT: apocrita_run
apocrita_run:
	sudo expect ./scripts/apocrita_run.sh \
 	${APOCRITA_USER} ${APOCRITA_PASSPHRASE} ${APOCRITA_USER_PASSWORD} ${AP_PRIVATE_KEY_PATH} \
 	${GIT_BRANCH} ${PROJECT_NAME} ${WANDB_API_TOKEN} ${START_SEED} ${END_SEED} ${RUN_NAME}

.SILENT: apocrita_clean_runs
apocrita_clean_runs:
	sudo expect ./scripts/apocrita_clean_runs.sh \
 	${APOCRITA_USER} ${APOCRITA_PASSPHRASE} ${APOCRITA_USER_PASSWORD} ${AP_PRIVATE_KEY_PATH} \
 	${GIT_BRANCH} ${PROJECT_NAME}


#NO_DEBUG     ?=
#NO_DOCSTRING ?=
#NO_DEBUG_CMD := $(if ${NO_DOCSTRING},-OO,$(if ${NO_DEBUG},-O,))
#PYTHON       ?= $(shell which python) ${NO_DEBUG_CMD}
#
#DOC_DIR        := ./docs
#DIST_DIR       := ./dist
#WHEELHOUSE_DIR := ./wheelhouse
#BENCHMARK_DIR  := ./benchmark
#SRC_DIR        := ./lzero
#RUNS_DIR       := ./runs
#
#RANGE_DIR       ?= .
#RANGE_TEST_DIR  := ${SRC_DIR}/${RANGE_DIR}
#RANGE_BENCH_DIR := ${BENCHMARK_DIR}/${RANGE_DIR}
#RANGE_SRC_DIR   := ${SRC_DIR}/${RANGE_DIR}
#
#CYTHON_FILES   := $(shell find ${SRC_DIR} -name '*.pyx')
#CYTHON_RELATED := \
#	$(addsuffix .c, $(basename ${CYTHON_FILES})) \
#	$(addsuffix .cpp, $(basename ${CYTHON_FILES})) \
#	$(addsuffix .h, $(basename ${CYTHON_FILES})) \
#
#COV_TYPES        ?= xml term-missing
#COMPILE_PLATFORM ?= manylinux_2_24_x86_64
#
#
#build:
#	$(PYTHON) setup.py build_ext --inplace \
#					$(if ${LINETRACE},--define CYTHON_TRACE,)
#
#zip:
#	$(PYTHON) -m build --sdist --outdir ${DIST_DIR}
#
#package:
#	$(PYTHON) -m build --sdist --wheel --outdir ${DIST_DIR}
#	for whl in `ls ${DIST_DIR}/*.whl`; do \
#		auditwheel repair $$whl -w ${WHEELHOUSE_DIR} --plat ${COMPILE_PLATFORM} && \
#		cp `ls ${WHEELHOUSE_DIR}/*.whl` ${DIST_DIR} && \
#		rm -rf $$whl ${WHEELHOUSE_DIR}/* \
#  	; done
#
#clean:
#	rm -rf $(shell find ${SRC_DIR} -name '*.so') \
#			$(if ${CYTHON_RELATED},$(shell ls ${CYTHON_RELATED} 2> /dev/null),)
#	rm -rf ${DIST_DIR} ${WHEELHOUSE_DIR}
#
#test: unittest benchmark
#
#unittest:
#	$(PYTHON) -m pytest "${RANGE_TEST_DIR}" \
#		-sv -m unittest \
#		$(shell for type in ${COV_TYPES}; do echo "--cov-report=$$type"; done) \
#		--cov="${RANGE_SRC_DIR}" \
#		$(if ${MIN_COVERAGE},--cov-fail-under=${MIN_COVERAGE},) \
#		$(if ${WORKERS},-n ${WORKERS},)
#
#minitest:
#	$(PYTHON) -m pytest "${SRC_DIR}/mcts/tests/test_game_block.py" \
#		-sv -m unittest \
#		$(shell for type in ${COV_TYPES}; do echo "--cov-report=$$type"; done) \
#		--cov="${SRC_DIR}/mcts/tests/test_game_block.py" \
#		$(if ${MIN_COVERAGE},--cov-fail-under=${MIN_COVERAGE},) \
#		$(if ${WORKERS},-n ${WORKERS},)
#
#docs:
#	$(MAKE) -C "${DOC_DIR}" build
#pdocs:
#	$(MAKE) -C "${DOC_DIR}" prod


# IRIS run experiments
