#!/bin/bash

# 搜得都队
TEAM_NAME="sd3"
git lfs install
#make some dirs

rm -r model
mkdir model
# # download the model if you need
git clone https://www.modelscope.cn/models/buaadreamer/bge-reranker-v2-minicpm-layerwise && mv bge-reranker-v2-minicpm-layerwise model/bge-reranker-v2-minicpm-layerwise

# prepare dataset and unzip
rm -rf data
git clone https://www.modelscope.cn/datasets/issaccv/aiops2024-challenge-dataset.git data
cd data
mkdir origin_data
echo "Unzip files..."
unzip -q -O gb2312 director.zedx -d origin_data/director
unzip -q -O utf-8 emsplus.zedx -d origin_data/emsplus
unzip -q -O gb2312 rcp.zedx -d origin_data/rcp
unzip -q -O utf-8 umac.zedx -d origin_data/umac
mv origin_data/umac/documents/Namf_MP/zh-CN origin_data/umac/documents/Namf_MP/zh-cn
echo "Unzip Done"
cd ..
echo $PWD

chmod +x src/pipeline.sh

# build docker image
docker build -t "$TEAM_NAME" .
# run docker container with model and data volume with sub network
docker run --gpus=all -itd --network=host --name "$TEAM_NAME" -v $PWD/src:/app -v $PWD/model:/model -v $PWD/data:/data "$TEAM_NAME" 

docker wait "$TEAM_NAME"
# copy the output file from the container to the host
docker cp "$TEAM_NAME":/app/submit_result.jsonl answer.jsonl
# remove the container
docker stop "$TEAM_NAME" && docker rm "$TEAM_NAME"
