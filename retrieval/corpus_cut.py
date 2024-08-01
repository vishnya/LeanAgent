# take contents of leandojo_benchmark_4_downloaded/leandojo_benchmark_4/corpus.jsonl and save just the first line to leandojo_benchmark_4_downloaded/leandojo_benchmark_4/corpus_cut.jsonl
import os
import json

corpus_path = "leandojo_benchmark_4_downloaded/leandojo_benchmark_4/corpus.jsonl"
corpus_cut_path = "leandojo_benchmark_4_downloaded/leandojo_benchmark_4/corpus_cut.jsonl"
for line in open(corpus_path):
    data = json.loads(line)
    # save data into corpus_cut_path
    with open(corpus_cut_path, "w") as f:
        json.dump(data, f)
    break
