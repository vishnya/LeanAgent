import subprocess
import psutil
from tabulate import tabulate

def get_gpu_processes():
    gpu_processes = []

    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'],
            text=True
        )
        for line in result.strip().split('\n'):
            gpu_uuid, pid, gpu_memory_usage = line.split(', ')
            pid = int(pid)
            gpu_memory_usage = int(gpu_memory_usage)
            try:
                process = psutil.Process(pid)
                process_name = process.name()
                user = process.username()
                gpu_processes.append({
                    'GPU UUID': gpu_uuid,
                    'PID': pid,
                    'User': user,
                    'Process Name': process_name,
                    'GPU Memory Usage (MB)': gpu_memory_usage
                })
            except psutil.NoSuchProcess:
                continue
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")

    return gpu_processes

def get_gpu_info():
    gpu_info = {}
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader,nounits'],
            text=True
        )
        for line in result.strip().split('\n'):
            index, uuid, name = line.split(', ')
            gpu_info[uuid] = {'index': index, 'name': name}
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")

    return gpu_info

def main():
    gpu_processes = get_gpu_processes()
    gpu_info = get_gpu_info()

    for process in gpu_processes:
        gpu_details = gpu_info.get(process['GPU UUID'], {'index': 'Unknown', 'name': 'Unknown'})
        process['GPU Index'] = gpu_details['index']
        process['GPU Name'] = gpu_details['name']
    if gpu_processes:
        print(tabulate(gpu_processes, headers="keys"))
    else:
        print("No GPU processes found.")

if __name__ == "__main__":
    main()