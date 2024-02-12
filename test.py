import torch
model_weights = torch.zeros((13,))
step_weights = torch.zeros((13,))
for i, model_weight in enumerate(model_weights):
    if model_weight == 0:
        print(step_weights[i])
