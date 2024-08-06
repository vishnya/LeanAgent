#!/bin/bash

export RUNPOD_API_KEY="YVDWAN4T13S1IXQNDG89BAGRD34WNP24HGZWBS0Y"
export HUGGINGFACE_TOKEN="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"
export GITHUB_ACCESS_TOKEN="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"

POD_INFO=$(runpodctl create pod \
  --imageName "ak123321/leancopilot-compute:latest" \
  --name "LeanCopilot-Compute" \
  --gpuCount 1 \
  --gpuType "NVIDIA A40" \
  --volumeSize 20 \
  --containerDiskSize 20 \
  --ports "8888/http,22/tcp" \
  --env "RUNPOD_API_KEY=$RUNPOD_API_KEY","HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN","GITHUB_ACCESS_TOKEN=$GITHUB_ACCESS_TOKEN" \
  --vcpu 9 \
  --mem 48 \
  --secureCloud)

POD_ID=$(echo "$POD_INFO" | awk '{print $2}')

echo "Created pod with ID: $POD_ID"
echo "Waiting for pod to start..."

sleep 120

# TODO: save logs and terminate
read -p "Press enter to stop the pod"
echo "Stopping pod..."
runpodctl stop pod $POD_ID

# docker build -t ak123321/leancopilot-compute:latest . && docker push ak123321/leancopilot-compute:latest

# TODO: do this from DGX
# crontab -e
# 5 23 * * *  >> /path/to/logfile.log 2>&1
