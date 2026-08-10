import torch

def add_batch_dim(data):
    """
    Recursively adds a batch dimension to tensors in a (possibly nested) dictionary.
    
    Args:
        data (dict or tensor): Dictionary or tensor to process.
        
    Returns:
        dict or tensor: The input structure with batch dimensions added to all tensors.
    """
    if isinstance(data, dict):
        # Recursively process dictionaries
        return {key: add_batch_dim(value) for key, value in data.items()}
    elif isinstance(data, torch.Tensor):
        # Add batch dimension to tensors
        return data.unsqueeze(0)
    else:
        # Return the item as is for non-tensors
        return data
