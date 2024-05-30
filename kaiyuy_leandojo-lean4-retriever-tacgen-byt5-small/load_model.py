from safetensors import safe_open
import os

print(os.getcwd())
tensors = {}
with safe_open("kaiyuy_leandojo-lean4-retriever-tacgen-byt5-small/model.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k) # loads the full tensor given a key
print(tensors)


