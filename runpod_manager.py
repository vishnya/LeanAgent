import requests
import time
import os

RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
API_URL = "https://api.runpod.io/v2"

def create_and_run_pod():
    url = f"{API_URL}/pod/create"
    # TODO: try to find most cost-effective settings later
    payload = {
        "name": "LeanCopilot-Compute",
        "imageName": "ak123321/leancopilot-compute:latest",
        "podType": "GPU",
        "gpuCount": 1,
        "volumeInGb": 20,
        "containerDiskInGb": 20,
        "gpuTypeId": "NVIDIA A40",
        "cloudType": "ALL",  # Most cost-effective option
        "startJupyterLab": False,
        "startSSH": True,
        "dockerArgs": "--shm-size=1g",
        "ports": "8888/http,22/tcp",
        "volumeMountPath": "/workspace",
        "env": [
            # TODO: add more?
            {"key": "HUGGINGFACE_TOKEN", "value": os.getenv('HUGGINGFACE_TOKEN')}
        ],
        "minVcpuCount": 9,
        "minMemoryInGb": 48,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    response = requests.post(url, json=payload, headers=headers)
    pod_info = response.json()

    if 'id' not in pod_info:
        raise Exception(f"Failed to create pod: {pod_info}")
    
    pod_id = pod_info['id']
    print(f"Created pod with ID: {pod_id}")

    # Wait for the pod to be ready
    while True:
        status_url = f"{API_URL}/pod/{pod_id}"
        status_response = requests.get(status_url, headers=headers)
        status = status_response.json()['status']
        if status == 'RUNNING':
            break
        time.sleep(10)

    print("Pod is now running")
    # compute_server.py will start automatically due to the CMD in Dockerfile
    print("compute_server.py should now be running in the pod")
    
    return pod_id

def terminate_pod(pod_id):
    url = f"{API_URL}/pod/{pod_id}/terminate"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    response = requests.post(url, headers=headers)
    return response.json()

def main():
    try:
        pod_id = create_and_run_pod()
    finally:
        if pod_id:
            print("Terminating pod...")
            terminate_result = terminate_pod(pod_id)
            print(f"Pod termination result: {terminate_result}")

if __name__ == "__main__":
    main()

# docker build -t ak123321/leancopilot-compute:latest .
# docker push ak123321/leancopilot-compute:latest

# export RUNPOD_API_KEY="YVDWAN4T13S1IXQNDG89BAGRD34WNP24HGZWBS0Y"
# export HUGGINGFACE_TOKEN="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"
# python runpod_manager.py

# TODO: run at 11:55 PM daily
# crontab -e
# 55 23 * * * python /home/adarsh/ReProver/runpod_manager.py
# 5 23 * * * /path/to/your/run_leancopilot.sh >> /path/to/logfile.log 2>&1