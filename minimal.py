import torch
import copy
from retrieval.model import PremiseRetriever
from loguru import logger

model_checkpoint_path = "leandojo-pl-ckpts/new_retriever.ckpt"
# model_checkpoint_path = "lightning_logs_failing_benchmark2/epoch=0-Recall@10_val=67.10.ckpt"
# optimizer_states_path = 'leandojo-pl-ckpts/retriever_random.ckpt/checkpoint/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt'
try:
    state_dict = torch.load(model_checkpoint_path)
    state_dict["pytorch-lightning_version"] = "0.0.0"
    state_dict['global_step'] = None
    state_dict['epoch'] = 0 # TODO: change
    #     {'fit_loop': {'state_dict': {}, 'epoch_loop.state_dict': {'_batches_that_stepped': 1106}, 'epoch_loop.batch_progress': {'total': {'ready': 1106, 'completed': 1106, 'started': 1106, 'processed': 1106}, 'current': {'ready': 1106, 'completed': 1106, 'started': 1106, 'processed': 1106}, 'is_last_batch': True}, 'epoch_loop.scheduler_progress': {'total': {'ready': 1106, 'completed': 1106}, 'current': {'ready': 1106, 'completed': 1106}}, 'epoch_loop.automatic_optimization.state_dict': {}, 'epoch_loop.automatic_optimization.optim_progress': {'optimizer': {'step': {'total': {'ready': 1106, 'completed': 1106}, 'current': {'ready': 1106, 'completed': 1106}}, 'zero_grad': {'total': {'ready': 1106, 'completed': 1106, 'started': 1106}, 'current': {'ready': 1106, 'completed': 1106, 'started': 1106}}}}, 'epoch_loop.manual_optimization.state_dict': {}, 'epoch_loop.manual_optimization.optim_step_progress': {'total': {'ready': 0, 'completed': 0}, 'current': {'ready': 0, 'completed': 0}}, 'epoch_loop.val_loop.state_dict': {}, 'epoch_loop.val_loop.batch_progress': {'total': {'ready': 1, 'completed': 1, 'started': 1, 'processed': 1}, 'current': {'ready': 1, 'completed': 1, 'started': 1, 'processed': 1}, 'is_last_batch': True}, 'epoch_progress': {'total': {'ready': 1, 'completed': 0, 'started': 1, 'processed': 1}, 'current': {'ready': 1, 'completed': 0, 'started': 1, 'processed': 1}}}, 'validate_loop': {'state_dict': {}, 'batch_progress': {'total': {'ready': 0, 'completed': 0, 'started': 0, 'processed': 0}, 'current': {'ready': 0, 'completed': 0, 'started': 0, 'processed': 0}, 'is_last_batch': False}}, 'test_loop': {'state_dict': {}, 'batch_progress': {'total': {'ready': 0, 'completed': 0, 'started': 0, 'processed': 0}, 'current': {'ready': 0, 'completed': 0, 'started': 0, 'processed': 0}, 'is_last_batch': False}}, 'predict_loop': {'state_dict': {}, 'batch_progress': {'total': {'ready': 0, 'completed': 0, 'started': 0, 'processed': 0}, 'current': {'ready': 0, 'completed': 0, 'started': 0, 'processed': 0}}}}
    # ipdb> state_dict["loops"]["predict_loop"]
    # {'state_dict': {}, 'batch_progress': {'total': {'ready': 0, 'completed': 0, 'started': 0, 'processed': 0}, 'current': {'ready': 0, 'completed': 0, 'started': 0, 'processed': 0}}}
    state_dict["loops"] = {}
    state_dict["loops"]["predict_loop"] = {'state_dict': {}, 'batch_progress': {'total': {'ready': 0, 'completed': 0, 'started': 0, 'processed': 0}, 'current': {'ready': 0, 'completed': 0, 'started': 0, 'processed': 0}}}
    state_dict['state_dict'] = state_dict
    # import ipdb; ipdb.set_trace()
    inner_dict = copy.deepcopy(state_dict['state_dict'])
    # Remove keys from the inner dictionary
    keys_to_remove = ['pytorch-lightning_version', 'global_step', 'epoch', 'loops', 'state_dict']
    for key in keys_to_remove:
        if key in inner_dict:
            del inner_dict[key]
    state_dict['state_dict'] = inner_dict

    # print("loading the optimizer states")

    # TODO: udpate any of these settings like scheduler if needed
    
    # optimizer_states = torch.load(optimizer_states_path)

    if not torch.cuda.is_available():
        logger.warning("Indexing the corpus using CPU can be very slow.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # TODO: reduce repetition in code like this
    config = {
        "model_name": "kaiyuy/leandojo-lean4-retriever-byt5-small",
        "lr": 1e-3,
        "warmup_steps": 1000,
        "max_seq_len": 512,
        "num_retrieved": 100,
    }

    torch.save(state_dict, model_checkpoint_path)

    print("Checkpoint loaded successfully!")

    model = PremiseRetriever.load(
        model_checkpoint_path, device, freeze=False, config=config
    )

    model.load_state_dict(state_dict['state_dict'])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    lambda1 = lambda epoch: 0.95 ** epoch  # Decays the learning rate each epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    # Serialize the dummy optimizer's state
    optimizer_state_dict = optimizer.state_dict()
    optimizer_state_dict['param_groups'][0]['params'] = list(range(len(optimizer_state_dict['param_groups'][0]['params'])))
    
    # import ipdb; ipdb.set_trace()

    # Add the dummy optimizer state to the checkpoint
    state_dict['optimizer_states'] = [optimizer_state_dict]  # Adjust key as needed based on your training script
    # inside list: {'state': {}, 'param_groups': [{'lr': 0.0001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False, 'maximize': False, 'foreach': None, 'capturable': False, 'differentiable': False, 'fused': None, 'params': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]}]}

    scheduler_state_dict = scheduler.state_dict()
    state_dict['lr_schedulers'] = [scheduler_state_dict]
    # inside list: {'base_lrs': [0.0001], 'last_epoch': 0, 'verbose': False, '_step_count': 1, '_get_lr_called_within_step': False, '_last_lr': [0.0001], 'lr_lambdas': [None]}

    # complete_checkpoint = {
    #     'state_dict': state_dict['state_dict'],
    #     'optimizer_states': optimizer_states,
    #     "pytorch-lightning_version": state_dict["pytorch-lightning_version"],
    #     "global_step": state_dict["global_step"],
    #     "epoch": state_dict["epoch"],
    #     "state_dict": state_dict["state_dict"]
    # }

    # state_dict['optimizer_states'] = optimizer_states["optimizer_state_dict"]

    # del state_dict['state_dict']['callbacks']
    # del state_dict['state_dict']['loops']
    # del state_dict['state_dict']['legacy_pytorch-lightning_version']
    # state_dict['callbacks'] = None
    # state_dict['loops'] = {}
    # state_dict['legacy_pytorch-lightning_version'] = None

    torch.save(state_dict, model_checkpoint_path)
    # torch.save(complete_checkpoint, model_checkpoint_path)
    print("Checkpoint loaded successfully with optimizers!")
except Exception as e:
    print(f"Failed to load checkpoint: {e}")
