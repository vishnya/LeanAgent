# change into /raid/adarsh/datasets_new/PutnamBench_82cb2b375c412582c35e759f991d3e6aac977815/random
import os
import sys
import json

os.chdir("/raid/adarsh/datasets_new/PutnamBench_82cb2b375c412582c35e759f991d3e6aac977815/random")
# load train.json
with open("train.json", "r") as f:
    train = json.load(f)

for line in train:
    if "putnam" in line["file_path"].lower():
        print(line)
        break
