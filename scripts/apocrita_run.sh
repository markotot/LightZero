#!/usr/bin/expect

set APOC_USERNAME [lindex $argv 0]
set APOC_PASSPHRASE [lindex $argv 1];
set APOC_PASSWORD [lindex $argv 2];
set APOC_PRIVATE_KEY [lindex $argv 3];

set GIT_BRANCH [lindex $argv 4];
set PROJECT_NAME [lindex $argv 5];
set WANDB_API_KEY [lindex $argv 6];

set START_SEED [lindex $argv 7];
set END_SEED [lindex $argv 8];
set RUN_NAME [lindex $argv 9];
set ENV_NAME [lindex $argv 10];


set PULL_JOB_PARAMS "-N ${RUN_NAME}-Pull -v JOB_TYPE=\"pull_git\",RUN_NAME=$RUN_NAME,GIT_BRANCH=$GIT_BRANCH $PROJECT_NAME/scripts/submit_experiment_job.sh"
set EXPERIMENT_JOB_PARAMS "-hold_jid ${RUN_NAME}-Pull -N ${RUN_NAME}-Experiment -v JOB_TYPE=\"run_experiment\",RUN_NAME=$RUN_NAME,START_SEED=$START_SEED,END_SEED=$END_SEED,ENV_NAME=$ENV_NAME,GIT_BRANCH=$GIT_BRANCH,WANDB_API_KEY=$WANDB_API_KEY $PROJECT_NAME/scripts/submit_experiment_job.sh"

spawn ssh -i $APOC_PRIVATE_KEY $APOC_USERNAME@login.hpc.qmul.ac.uk \
 "
  source ../../../../../etc/bashrc; \
  rm myenvs; \
  echo WANDB_API_KEY=$WANDB_API_KEY > myenvs; \

  qsub $PULL_JOB_PARAMS ; \
  qsub $EXPERIMENT_JOB_PARAMS ; \
 "
expect "Enter passphrase for key '$APOC_PRIVATE_KEY':"
send "$APOC_PASSPHRASE\r"
expect "$APOC_USERNAME@login.hpc.qmul.ac.uk's password"
send "$APOC_PASSWORD\r"
interact
