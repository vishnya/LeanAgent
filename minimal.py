import torch
import copy

model_checkpoint_path = "leandojo-pl-ckpts/new_retriever.ckpt"
# optimizer_states_path = 'leandojo-pl-ckpts/retriever_random.ckpt/checkpoint/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt'
try:
    state_dict = torch.load(model_checkpoint_path)
    state_dict["pytorch-lightning_version"] = "0.0.0"
    state_dict['global_step'] = None
    state_dict['epoch'] = 5 # TODO: change
    state_dict['state_dict'] = state_dict
    # import ipdb; ipdb.set_trace()
    inner_dict = copy.deepcopy(state_dict['state_dict'])
    # Remove keys from the inner dictionary
    keys_to_remove = ['pytorch-lightning_version', 'global_step', 'epoch', 'state_dict']
    for key in keys_to_remove:
        if key in inner_dict:
            del inner_dict[key]
    state_dict['state_dict'] = inner_dict

    # print("loading the optimizer states")
    
    # optimizer_states = torch.load(optimizer_states_path)

    # import ipdb; ipdb.set_trace()

    # complete_checkpoint = {
    #     'state_dict': state_dict['state_dict'],
    #     'optimizer_states': optimizer_states,
    #     "pytorch-lightning_version": state_dict["pytorch-lightning_version"],
    #     "global_step": state_dict["global_step"],
    #     "epoch": state_dict["epoch"],
    #     "state_dict": state_dict["state_dict"]
    # }

    # del state_dict['state_dict']['callbacks']
    # del state_dict['state_dict']['loops']
    # del state_dict['state_dict']['legacy_pytorch-lightning_version']
    # state_dict['callbacks'] = None
    # state_dict['loops'] = {}
    # state_dict['legacy_pytorch-lightning_version'] = None

    torch.save(state_dict, model_checkpoint_path)
    # torch.save(complete_checkpoint, model_checkpoint_path)
    print("Checkpoint loaded successfully!")
except Exception as e:
    print(f"Failed to load checkpoint: {e}")
