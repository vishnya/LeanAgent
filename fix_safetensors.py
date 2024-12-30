from loguru import logger
import torch
from generator.model import RetrievalAugmentedGenerator, FixedTacticGenerator

model_checkpoint_path = "/data/yingzi_ma/lean_project/checkpoints_PT_full_merge_each_time_ewc/mathlib4_29dcec074de168ac2bf835a77ef68bbe069194c5.ckpt"

config = {
    "model_name": "kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small",
    "lr": 1e-3,
    "warmup_steps": 1000,
    "num_beams": 5,
    "eval_num_retrieved": 10,
    "eval_num_workers": 1,
    "eval_num_gpus": 1,  # TODO: change for GPU
    "eval_num_theorems": 100,
    "max_inp_seq_len": 512,
    "max_oup_seq_len": 128,
    "ret_ckpt_path": model_checkpoint_path,
}

ckpt_path = "/data/yingzi_ma/lean_project/checkpoints_PT_full_merge_each_time_ewc/mathlib4_29dcec074de168ac2bf835a77ef68bbe069194c5.ckpt"
if not torch.cuda.is_available():
    logger.warning("Indexing the corpus using CPU can be very slow.")
    device = torch.device("cpu")
else:
    device = torch.device("cuda")

try:
    model = RetrievalAugmentedGenerator.load(
        ckpt_path,
        device,
        freeze=False,
        config=config
    )
except Exception as e:
    logger.exception("Error loading model")
    raise