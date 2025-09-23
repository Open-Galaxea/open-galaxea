#!/bin/bash

set -e

TASK=
for arg in "$@"
do
    if [[ $arg == task=* ]]; then
        TASK="${arg#task=}"
        break
    fi
done

if [[ -z "$TASK" ]]; then
    echo "Error: 'Task' must be provided."
    exit 1
fi

PROJECT="${TASK%%/*}"
REMAINDER="${TASK#*/}"
GROUP="${REMAINDER%%/*}"
NAME="${REMAINDER#*/}"

PROJECT="logger.wandb.project=${PROJECT}"
GROUP="logger.wandb.group=${GROUP}"
NAME="logger.wandb.name=${NAME}_\${now:%Y-%m-%d}_\${now:%H-%M-%S}"
HYDRA_RUN_DIR="hydra.run.dir=\"\${oc.env:GALAXEA_DP_WORK_DIR}/${TASK}/\${now:%Y-%m-%d}_\${now:%H-%M-%S}\""

HYDRA_FULL_ERROR=1 python src/train.py "$@" $HYDRA_RUN_DIR $PROJECT $GROUP $NAME