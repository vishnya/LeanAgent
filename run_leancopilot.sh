#!/bin/bash

export RUNPOD_API_KEY="YVDWAN4T13S1IXQNDG89BAGRD34WNP24HGZWBS0Y"
export HUGGINGFACE_TOKEN="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"
export GITHUB_ACCESS_TOKEN="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"

# TODO: do we need env if we already export?
POD_ID=$(runpodctl create pod \
  --imageName "ak123321/leancopilot-compute:latest" \
  --name "LeanCopilot-Compute" \
  --gpuCount 1 \
  --gpuType "NVIDIA A40" \
  --volumeSize 20 \
  --containerDiskSize 20 \
  --ports "8888/http,22/tcp" \
  --env "HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN" \
  --vcpu 9 \
  --mem 48 \
  --secureCloud)

POD_ID=$(echo "$POD_ID" | awk '{print $2}' | tr -d '"')

echo "Created pod with ID: $POD_ID"

while true; do
    STATUS=$(runpodctl get pod "$POD_ID" | awk 'NR==2 {print $NF}')
    if [ "$STATUS" == "RUNNING" ]; then
        break
    fi
    echo "Current status: $STATUS. Waiting..."
    sleep 10
done

echo "Pod is now running and compute_server.py should be executing"

read -p "Press enter to stop the pod"
runpodctl stop pod $POD_ID
# TODO: terminate pod and save logs

# TODO: do this from DGX
# crontab -e
# 5 23 * * *  >> /path/to/logfile.log 2>&1
