#!/bin/bash

### 搜的都队
TEAM_NAME="sddd_easyrag"
git lfs install

### If it hasn't been run
# bash scripts/download.sh # download models
# bash scripts/process.sh # process zedx data

### build docker image
docker build -t "$TEAM_NAME" .
### run docker container with model and data volume with sub network
docker run --gpus=all -itd --network=host --name "$TEAM_NAME" -v $PWD/src:/app -v $PWD/models:/models -v $PWD/data:/data "$TEAM_NAME"

docker wait "$TEAM_NAME"
### copy the output file from the container to the host
docker cp "$TEAM_NAME":/app/submit_result.jsonl answer.jsonl
### remove the container
docker stop "$TEAM_NAME" && docker rm "$TEAM_NAME"
