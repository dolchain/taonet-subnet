uids = [6, 10, 11]
miner_uids = {6: [7,8,9], 10: [], 11: [7, 9]}
win_rate={6:1.0, 10: 0.3417, 11: 0.1583}
import torch
model_weights = torch.zeros((13,))
for vali_uid in miner_uids:
    for miner_uid in miner_uids[vali_uid]:
        model_weights[miner_uid] += win_rate[vali_uid]
print(model_weights)
step_weights = torch.softmax(model_weights / 1, dim=0)
print(step_weights)
# Update weights based on moving average.
new_weights = torch.zeros((13,))
for i, model_weight in enumerate(model_weights):
    if model_weight:
        new_weights[i] = step_weights[i]

new_weights /= new_weights.sum()
print(new_weights)
new_weights = new_weights.nan_to_num(0.0)
print(new_weights)