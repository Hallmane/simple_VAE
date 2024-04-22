import torch

def print_shape(module, input, output):
    print(f"{module.__class__.__name__} output shape: {output.shape}")
