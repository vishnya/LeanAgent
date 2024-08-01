# save new json files with only 5% of the data in pfr_benchmark_2/random

import json


data = json.load(open("new_version_test2_benchmark/random/train.json"))
print(len(data))
data = data[: int(len(data) * 0.05)]
print(len(data))
json.dump(data, open("new_version_test2_benchmark/random5percent/train.json", "w"))
import ipdb; ipdb.set_trace()
data = json.load(open("new_version_test2_benchmark/random/val.json"))
print(len(data))
data = data[: int(len(data) * 0.05)]
print(len(data))
json.dump(data, open("new_version_test2_benchmark/random5percent/val.json", "w"))

data = json.load(open("new_version_test2_benchmark/random/test.json"))
print(len(data))
data = data[: int(len(data) * 0.05)]
print(len(data))
json.dump(data, open("new_version_test2_benchmark/random5percent/test.json", "w"))