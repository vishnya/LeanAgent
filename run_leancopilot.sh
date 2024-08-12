#!/bin/bash

export RUNPOD_API_KEY="YVDWAN4T13S1IXQNDG89BAGRD34WNP24HGZWBS0Y"
export HUGGINGFACE_TOKEN="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"
export GITHUB_ACCESS_TOKEN="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"
export MEGA_EMAIL="adarshk1234567@gmail.com"
export MEGA_PASSWORD="0879197abc"

# TODO: change image name later
POD_INFO=$(runpodctl create pod \
  --imageName "ak123321/leancopilot-compute:latest" \
  --name "LeanCopilot-Compute" \
  --gpuCount 1 \
  --gpuType "NVIDIA A40" \
  --volumeSize 30 \
  --containerDiskSize 30 \
  --ports "8888/http,22/tcp" \
  --env "RUNPOD_API_KEY=$RUNPOD_API_KEY","HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN","GITHUB_ACCESS_TOKEN=$GITHUB_ACCESS_TOKEN","MEGA_EMAIL=$MEGA_EMAIL","MEGA_PASSWORD=$MEGA_PASSWORD" \
  --vcpu 9 \
  --mem 48 \
  --secureCloud)

POD_ID=$(echo "$POD_INFO" | awk '{print $2}')

echo "Created pod with ID: $POD_ID"
echo "Waiting for pod to start..."

sleep 120

read -p "Press enter to stop the pod"
echo "Stopping pod..."
runpodctl stop pod $POD_ID

# docker build -t ak123321/leancopilot-compute:latest . && docker push ak123321/leancopilot-compute:latest
# docker build -t ak123321/leancopilot-compute-medium:latest . && docker push ak123321/leancopilot-compute-medium:latest

# crontab -e
# 5 23 * * *  >> /path/to/logfile.log 2>&1
