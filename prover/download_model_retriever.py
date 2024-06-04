from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import torch

def download_model(model_name: str, ckpt_model_path: str):
    try:
        print(f"Downloading tokenizer for {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Tokenizer downloaded")

        print(f"Downloading model for {model_name}")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("Model downloaded successfully.")

        print(model.state_dict().keys())
        print(type(model.state_dict()))

        state_dict = model.state_dict()
        print(f"before size: {len(state_dict)}")
        new_dict = {}
        for key, value in state_dict.items():
            parts = key.split('.')
            if (parts[0] == 'encoder') and (parts[1] == 'decoder'):
                parts[1] = 'encoder'
                new_key = '.'.join(parts)
                new_key = "retriever." + new_key
                new_dict[new_key] = value
            else:
                new_dict['.'.join(parts)] = value
        state_dict = new_dict
        print(f"after size: {len(state_dict)}")
        state_dict['pytorch-lightning_version'] = "0.0.0"
        state_dict['global_step'] = None
        state_dict['epoch'] = None
        state_dict['state_dict'] = state_dict
        state_dict['callbacks'] = None
        state_dict['loops'] = {}
        state_dict['legacy_pytorch-lightning_version'] = None
        print(state_dict.keys())

        # Save the model in .ckpt format
        print(f"Saving model to {ckpt_model_path}")
        torch.save(state_dict, ckpt_model_path)
        print(f"Model saved in .ckpt format at {ckpt_model_path}")
        print(f"Model and tokenizer saved locally in ./{model_name.replace('/', '_')}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    model_name = "kaiyuy/leandojo-lean4-retriever-byt5-small"
    ckpt_model_path = '<DIR>/kaiyuy_leandojo-lean4-retriever-tacgen-byt5-small/model_lightning_retriever.ckpt' # Add your path here
    print(os.getcwd())
    download_model(model_name, ckpt_model_path)
