import GPUtil

def get_free_gpu_memory():
    gpus = GPUtil.getGPUs()
    free_memory_list = [gpu.memoryFree for gpu in gpus]
    return free_memory_list