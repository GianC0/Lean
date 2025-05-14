'''
UMI Loss Functions (adapted for Lean)

ACTION REQUIRED:
1. Populate this file with the content from src/loss.py of the UMI GitHub repository.
2. Adjust import statements to use relative imports for other UMI modules in this directory, e.g., `from .utils_lean import ...`.
'''

import torch
from torch import Tensor

# Attempt to import from utils_lean.py in the same directory
# This requires utils_lean.py to be correctly populated and weighted_corrcoef to be defined.
from .utils_lean import generate_weight, weighted_corrcoef

def mse_loss(output_data: Tensor, target_data: Tensor):
    # This is an example, ensure it matches the original UMI loss
    diff = output_data - target_data
    square_diff = torch.pow(diff, 2)
    return square_diff.mean()

def ic_loss(output_data: Tensor, target_data: Tensor, device, method=None):
    # This is an example, ensure it matches the original UMI loss
    # and that weighted_corrcoef is correctly implemented and imported.
    if weighted_corrcoef is None or generate_weight is None:
        print("WARNING: weighted_corrcoef or generate_weight not available in loss_lean.py. IC Loss will not work.")
        return torch.tensor(0.0) # Return a dummy value

    weight = generate_weight(output_data.size(0), method=method).to(device)
    ic = weighted_corrcoef(output_data, target_data, weight) # This will fail if not implemented
    return -ic
