import torch
import argparse
from loguru import logger

from retrieval.model import PremiseRetriever


def convert(src: str, dst: str) -> None:
    device = torch.device("cpu")
    config = {
        "model_name": "kaiyuy/leandojo-lean4-retriever-byt5-small",
        "lr": 1e-3,
        "warmup_steps": 1000,
        "max_seq_len": 512,
        "num_retrieved": 100,
    }
    model = PremiseRetriever.load(src, device, freeze=True, config=config)
    model.encoder.save_pretrained(dst)
    model.tokenizer.save_pretrained(dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    args = parser.parse_args()
    logger.info(args)

    logger.info(f"Loading the model from {args.src}")
    convert(args.model_type, args.src, args.dst)
    logger.info(f"The model saved in Hugging Face format to {args.dst}")


if __name__ == "__main__":
    main()

# python scripts/convert_checkpoint.py --src /raid/adarsh/checkpoints/mathlib4_29dcec074de168ac2bf835a77ef68bbe069194c5.ckpt --dst /home/adarsh/ReProver/leandojo-lean4-retriever-byt5-small