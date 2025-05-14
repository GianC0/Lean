'''
UMI Utilities (adapted for Lean)

ACTION REQUIRED:
1. Populate this file with the content from src/utils.py of the UMI GitHub repository.
2. Ensure all necessary functions, including `weighted_corrcoef`, are present.
3. Adjust any imports to be relative if they refer to other UMI modules within this directory (e.g., `from .other_umi_module import ...`).
'''

# Placeholder for UMI utility functions
# Make sure to include generate_weight, weighted_corrcoef, etc.

import torch

def generate_weight(stock_num, method=None):
    # This is an example, ensure it matches the original UMI util
    if method is None:
        return torch.ones((stock_num,)) / stock_num
    weight_list = []
    one_decile = stock_num // 10
    if method == "exp_decay":
        for j in range(10):
            if j < 9:
                weight_list += [0.9 ** j] * one_decile
            else:
                weight_list += [0.9 ** j] * (stock_num - one_decile * 9)
    else: # Example: linear decay, original might differ
        for j in range(10):
            if j < 9:
                weight_list += [(10 - j) / 10] * one_decile
            else:
                weight_list += [(10 - j) / 10] * (stock_num - one_decile * 9)
    weight = torch.FloatTensor(weight_list)
    weight /= weight.sum()
    assert len(weight) == stock_num
    return weight

# def weighted_corrcoef(output_data, target_data, weight):
#     # !!! IMPORTANT: Implement this function from the UMI repository !!!
#     # Placeholder implementation, likely incorrect:
#     # output_mean = torch.mean(output_data)
#     # target_mean = torch.mean(target_data)
#     # cov = torch.sum(weight * (output_data - output_mean) * (target_data - target_mean))
#     # std_output = torch.sqrt(torch.sum(weight * (output_data - output_mean)**2))
#     # std_target = torch.sqrt(torch.sum(weight * (target_data - target_mean)**2))
#     # if std_output * std_target == 0:
#     #     return torch.tensor(0.0)
#     # return cov / (std_output * std_target)
    pass
