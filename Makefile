GIT_BRANCH = main
PROJECT_NAME = LightZero

#APOCRITA
AP_PRIVATE_KEY_PATH = ~/Apocrita/apocrita.ssh
APOCRITA_USER = acw549

#EXPERIMENT CONFIG
NUM_SEEDS = 10
ENV_NAME = "BreakoutNoFrameskip-v4"
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
 	${GIT_BRANCH} ${PROJECT_NAME} ${WANDB_API_TOKEN} ${NUM_SEEDS} ${RUN_NAME} ${ENV_NAME}

.SILENT: apocrita_batch_run
apocrita_batch_run:
	sudo expect ./scripts/apocrita_batch_run.sh \
 	${APOCRITA_USER} ${APOCRITA_PASSPHRASE} ${APOCRITA_USER_PASSWORD} ${AP_PRIVATE_KEY_PATH} \
 	${GIT_BRANCH} ${PROJECT_NAME} ${WANDB_API_TOKEN} ${NUM_SEEDS} ${RUN_NAME}

 .SILENT: apocrita_download_checkpoints
apocrita_download_checkpoints:
	sudo expect ./scripts/apocrita_download_checkpoints.sh \
 	${APOCRITA_USER} ${APOCRITA_PASSPHRASE} ${APOCRITA_USER_PASSWORD} ${AP_PRIVATE_KEY_PATH} \
 	${GIT_BRANCH} ${PROJECT_NAME} ${WANDB_API_TOKEN} ${RUN_NAME}

.SILENT: apocrita_clean_runs
apocrita_clean_runs:
	sudo expect ./scripts/apocrita_clean_runs.sh \
 	${APOCRITA_USER} ${APOCRITA_PASSPHRASE} ${APOCRITA_USER_PASSWORD} ${AP_PRIVATE_KEY_PATH} \
 	${GIT_BRANCH} ${PROJECT_NAME}

.SILENT: apocrita_qstat
apocrita_qstat:
	sudo expect ./scripts/apocrita_qstat.sh \
 	${APOCRITA_USER} ${APOCRITA_PASSPHRASE} ${APOCRITA_USER_PASSWORD} ${AP_PRIVATE_KEY_PATH}
