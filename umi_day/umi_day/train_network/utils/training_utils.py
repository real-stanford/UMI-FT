from torch import nn

def get_gradient_norm(model: nn.Module):
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.norm(2).item()  # L2 norm
            total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    return total_norm
