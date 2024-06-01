from collections import OrderedDict
import torch

# Load the checkpoint
checkpoint_path = "/raid/adarsh/kaiyuy_leandojo-lean4-retriever-tacgen-byt5-small/model.ckpt"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# create new checkpoint
modified_checkpoint = OrderedDict()
modified_checkpoint['state_dict'] = OrderedDict()

# prefix keys in state_dict
state_dict = checkpoint['state_dict'].copy()
for k, v in state_dict.items():
    k = f'model.{k}'
    modified_checkpoint['state_dict'][k] = v

# add missing keys
modified_checkpoint['pytorch-lightning_version'] = '0.0.0'
# modified_checkpoint['global_step'] = None
# modified_checkpoint['epoch'] = None
# modified_checkpoint['state_dict']['predict_loop'] = None

# save
modified_checkpoint_path = "/raid/adarsh/kaiyuy_leandojo-lean4-retriever-tacgen-byt5-small/model_edited.ckpt"
torch.save(modified_checkpoint, modified_checkpoint_path)
