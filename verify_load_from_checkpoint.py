import torch
from retrieval.model import PremiseRetriever

# model_checkpoint_path = '/raid/adarsh/checkpoints/mathlib4_29dcec074de168ac2bf835a77ef68bbe069194c5.ckpt'
model_checkpoint_path = '/raid/adarsh/checkpoints/SciLean_2d1a4e79acf3a256ba2ec8ac2848d13f219b9684_lambda_0.0_epoch=0-Recall@10_val=68.15.ckpt'
device = torch.device("cuda")
config = {
            "model_name": "kaiyuy/leandojo-lean4-retriever-byt5-small",
            "lr": 1e-3,
            "warmup_steps": 1000,
            "max_seq_len": 512,
            "num_retrieved": 100,
        }

model = PremiseRetriever.load_hf("kaiyuy/leandojo-lean4-retriever-byt5-small", 512, device)
# print(model)
import ipdb; ipdb.set_trace()
print(model.encoder.encoder.block[0].layer[0].SelfAttention.q.weight)
print(model.encoder.encoder.block[1].layer[0].SelfAttention.q.weight)
print(model.encoder.encoder.block[1].layer[1].DenseReluDense.wo.weight)
print(model.encoder.encoder.block[2].layer[0].SelfAttention.v.weight)
tokenizer = model.tokenizer

model = PremiseRetriever.load(
            model_checkpoint_path, device, freeze=False, config=config
        )
# print(model)
print(model.encoder.encoder.block[0].layer[0].SelfAttention.q.weight)
print(model.encoder.encoder.block[1].layer[0].SelfAttention.q.weight)
print(model.encoder.encoder.block[1].layer[1].DenseReluDense.wo.weight)
print(model.encoder.encoder.block[2].layer[0].SelfAttention.v.weight)
print(tokenizer == model.tokenizer) # False

# import ipdb; ipdb.set_trace()

# print with hf and without hf SAME MODEL WITH MATHLIB, SCILEAN
# print state dicts THEY ARE DIFFERENT!
# what happens if remove config BOTH MODEL AND WEIGHTS SAME AS WITH CONFIG! 
# ask claude DONE




# Parameter containing:
# tensor([[ 0.0240, -0.0042,  0.0028,  ...,  0.0605,  0.0649,  0.0165],
#         [-0.0170,  0.0205,  0.0219,  ...,  0.0579,  0.0806, -0.0109],
#         [-0.0242,  0.0503,  0.0596,  ..., -0.0498, -0.0393,  0.0013],
#         ...,
#         [ 0.1138, -0.1855,  0.0016,  ...,  0.0033, -0.0275,  0.0160],
#         [-0.0232,  0.0099, -0.0649,  ...,  0.0143, -0.0020,  0.0281],
#         [ 0.0025, -0.0137, -0.0072,  ..., -0.0339, -0.0173,  0.0010]],
#        device='cuda:0', dtype=torch.bfloat16, requires_grad=True)



# Parameter containing:
# tensor([[-0.0031, -0.0286, -0.0345,  ...,  0.0669,  0.0220, -0.0093],
#         [-0.0708, -0.0330,  0.0649,  ...,  0.0247,  0.0080,  0.0262],
#         [-0.0336, -0.0197, -0.0003,  ..., -0.0357,  0.0511,  0.0270],
#         ...,
#         [ 0.0410, -0.1176, -0.0777,  ...,  0.0355,  0.0570,  0.0908],
#         [-0.0180,  0.0044,  0.0245,  ..., -0.0076,  0.0141, -0.0280],
#         [-0.0362,  0.0393,  0.0109,  ..., -0.0334, -0.0104, -0.0833]],
#        device='cuda:0', requires_grad=True)




# PremiseRetriever(
#   (encoder): T5EncoderModel(
#     (shared): Embedding(384, 1472)
#     (encoder): T5Stack(
#       (embed_tokens): Embedding(384, 1472)
#       (block): ModuleList(
#         (0): T5Block(
#           (layer): ModuleList(
#             (0): T5LayerSelfAttention(
#               (SelfAttention): T5Attention(
#                 (q): Linear(in_features=1472, out_features=384, bias=False)
#                 (k): Linear(in_features=1472, out_features=384, bias=False)
#                 (v): Linear(in_features=1472, out_features=384, bias=False)
#                 (o): Linear(in_features=384, out_features=1472, bias=False)
#                 (relative_attention_bias): Embedding(32, 6)
#               )
#               (layer_norm): T5LayerNorm()
#               (dropout): Dropout(p=0.1, inplace=False)
#             )
#             (1): T5LayerFF(
#               (DenseReluDense): T5DenseGatedActDense(
#                 (wi_0): Linear(in_features=1472, out_features=3584, bias=False)
#                 (wi_1): Linear(in_features=1472, out_features=3584, bias=False)
#                 (wo): Linear(in_features=3584, out_features=1472, bias=False)
#                 (dropout): Dropout(p=0.1, inplace=False)
#                 (act): NewGELUActivation()
#               )
#               (layer_norm): T5LayerNorm()
#               (dropout): Dropout(p=0.1, inplace=False)
#             )
#           )
#         )
#         (1-11): 11 x T5Block(
#           (layer): ModuleList(
#             (0): T5LayerSelfAttention(
#               (SelfAttention): T5Attention(
#                 (q): Linear(in_features=1472, out_features=384, bias=False)
#                 (k): Linear(in_features=1472, out_features=384, bias=False)
#                 (v): Linear(in_features=1472, out_features=384, bias=False)
#                 (o): Linear(in_features=384, out_features=1472, bias=False)
#               )
#               (layer_norm): T5LayerNorm()
#               (dropout): Dropout(p=0.1, inplace=False)
#             )
#             (1): T5LayerFF(
#               (DenseReluDense): T5DenseGatedActDense(
#                 (wi_0): Linear(in_features=1472, out_features=3584, bias=False)
#                 (wi_1): Linear(in_features=1472, out_features=3584, bias=False)
#                 (wo): Linear(in_features=3584, out_features=1472, bias=False)
#                 (dropout): Dropout(p=0.1, inplace=False)
#                 (act): NewGELUActivation()
#               )
#               (layer_norm): T5LayerNorm()
#               (dropout): Dropout(p=0.1, inplace=False)
#             )
#           )
#         )
#       )
#       (final_layer_norm): T5LayerNorm()
#       (dropout): Dropout(p=0.1, inplace=False)
#     )
#   )
# )







# Parameter containing:
# tensor([[ 0.0240, -0.0042,  0.0028,  ...,  0.0605,  0.0649,  0.0165],
#         [-0.0170,  0.0205,  0.0219,  ...,  0.0579,  0.0806, -0.0109],
#         [-0.0242,  0.0503,  0.0596,  ..., -0.0498, -0.0393,  0.0013],
#         ...,
#         [ 0.1138, -0.1855,  0.0016,  ...,  0.0033, -0.0275,  0.0160],
#         [-0.0232,  0.0099, -0.0649,  ...,  0.0143, -0.0020,  0.0281],
#         [ 0.0025, -0.0137, -0.0072,  ..., -0.0339, -0.0173,  0.0010]],
#        device='cuda:0', dtype=torch.bfloat16, requires_grad=True)



# Parameter containing:
# tensor([[-0.0620, -0.0089,  0.0143,  ...,  0.0383,  0.0608, -0.1270],
#         [ 0.0791,  0.0464,  0.0513,  ..., -0.0757, -0.1514, -0.0060],
#         [ 0.0378, -0.0059,  0.0674,  ..., -0.0347, -0.0240,  0.0203],
#         ...,
#         [ 0.0104,  0.0796, -0.0272,  ...,  0.0820,  0.1064,  0.0153],
#         [-0.0019, -0.0781, -0.0112,  ...,  0.0070,  0.0610,  0.0811],
#         [ 0.0265, -0.0420, -0.0208,  ..., -0.0444, -0.0281, -0.0137]],
#        device='cuda:0', dtype=torch.bfloat16, requires_grad=True)

# Parameter containing:
# tensor([[ 0.0688, -0.0879,  0.0972,  ..., -0.0142, -0.1699, -0.0199],
#         [ 0.4727, -0.1934,  0.3574,  ...,  0.3164,  0.3477,  0.0613],
#         [-0.0415, -0.0767, -0.0160,  ..., -0.1235,  0.2285,  0.1099],
#         ...,
#         [ 0.1846, -0.2471, -0.1069,  ...,  0.3340,  0.1543,  0.1992],
#         [-0.1201,  0.0913, -0.3262,  ..., -0.0835, -0.0183, -0.2070],
#         [-0.0457,  0.1128, -0.2002,  ..., -0.1040,  0.1387,  0.4629]],
#        device='cuda:0', dtype=torch.bfloat16, requires_grad=True)

# Parameter containing:
# tensor([[ 0.0977,  0.0491,  0.0491,  ..., -0.1177, -0.1973, -0.0432],
#         [ 0.0081, -0.3262,  0.1533,  ..., -0.0806, -0.1348,  0.1572],
#         [ 0.1738, -0.0187, -0.2891,  ..., -0.3477, -0.0679, -0.1924],
#         ...,
#         [ 0.1543,  0.2539, -0.0623,  ...,  0.0708,  0.2031,  0.4004],
#         [ 0.0884, -0.0811,  0.0894,  ..., -0.4629, -0.0142, -0.0903],
#         [ 0.1973, -0.0649,  0.3770,  ..., -0.1846, -0.0161, -0.0091]],
#        device='cuda:0', dtype=torch.bfloat16, requires_grad=True)