import torch.optim as optim
import estimator.stable_estimators as se 
import numpy as np
import torch

#--------------------------------------------------------------------------------------------
# Optimizer
#--------------------------------------------------------------------------------------------

def get_optimizer(model, optimizer_name='sgd', lr=1e-3, weight_decay=0.0):
    """Helper Funtion to get optimiuers

    Args:
        model
        optimizer_name (str, optional): Defaults to 'sgd'.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        weight_decay (float, optional): Weight Decay. Defaults to 0.0.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Unknown optimizer")
    

#--------------------------------------------------------------------------------------------
# Regularizer
#--------------------------------------------------------------------------------------------
    
def l1_regularization(model, lambda_l1=1e-4):
    """Standard Lasso Regression (L1)

    Args:
        model (torch.model): model that should be regularized
        lambda_l1 (float, optional): Lambda (Regularization Parameter) -> Hyperparameter Tuning. Defaults to 1e-4.

    Returns:
        float: sum of absolut weight values * lambda value 
    """
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return lambda_l1 * l1_norm

def l2_regularization(model, lambda_l2=1e-4):
    """Standard Ridge Regression (L2)

    Args:
        model (torch.model): model that should be regularized
        lambda_l1 (float, optional): Lambda (Regularization Parameter) -> Hyperparameter Tuning. Defaults to 1e-4.

    Returns:
        float: sum of absolut weight values * lambda value 
    """
    l2_norm = sum(torch.sum(p ** 2) for p in model.parameters())
    return lambda_l2 * l2_norm


def hill_regularizer(model, k_ratio=0.1, reduction="sum", eps=1e-12):
    """First Implemetation of Hill Regularization (Hill Regularizer see Nolan - Univariate Stable Distributions)

    Args:
        model 
        k_ratio (float, optional): Values that should be selected. Defaults to 0.1.
        reduction (str, optional): Sum vs mean of alpha values. Defaults to "sum".
        eps (float, optional): Small number for mumeric stability. Defaults to 1e-12.

    Returns:
        float: mean/sum of alphas that will be added to loss function
    """


    losses = []

    for name, param in model.named_parameters():
        if "weight" not in name:
            continue

        x = param.view(-1).abs() + eps   

        x_sorted, _ = torch.sort(x, descending=True)

        k = max(5, int(len(x_sorted) * k_ratio))
        topk = x_sorted[:k]

        logs = torch.log(topk + eps)
        alpha_hat = 1.0 / (logs.mean() - logs[-1] + eps)

        losses.append(alpha_hat)

    stacked = torch.stack(losses)

    if reduction == "mean":
        return stacked.mean()
    else:
        return stacked.sum()



def hill_regularizer_weighted(model, k_ratio=0.1, reduction="sum", eps=1e-12):

    losses = []

    for i, (name, param) in enumerate(model.named_parameters()):
        if "weight" not in name:
            continue

        x = param.view(-1).abs() + eps
        x_sorted, _ = torch.sort(x, descending=True)
        k = max(5, int(len(x_sorted) * k_ratio))
        topk = x_sorted[:k]

        logs = torch.log(topk + eps)
        alpha_hat = 1.0 / (logs.mean() - logs[-1] + eps)

      
        losses.append(alpha_hat)

    stacked = torch.stack(losses)
    if reduction == "mean":
        return stacked.mean()
    else:
        return stacked.sum()


def parabolic_hill_single(param, k_ratio=0.1, eps=1e-12):
    """
    Parabolic Hill regularizer for a single weight tensor.

    Args:
        param (torch.Tensor): weight tensor of a single layer
        k_ratio (float): fraction of largest weights used for tail estimation
        eps (float): numerical stability constant

    Returns:
        torch.Tensor: scalar regularization loss for this layer
    """
    x = param.view(-1).abs() + eps
    x_sorted, _ = torch.sort(x, descending=True)

    k = max(5, int(len(x_sorted) * k_ratio))
    topk = x_sorted[:k]

    logs = torch.log(topk + eps)
    alpha_hat = 1.0 / (logs.mean() - logs[-1] + eps)

    return (1.0 - alpha_hat) ** 2

def parabolic_hill(model, k_ratio=0.1, reduction="sum", eps=1e-12):
    losses = []

    for name, param in model.named_parameters():
        if "weight" not in name:
            continue

        losses.append(
            parabolic_hill_single(param, k_ratio=k_ratio, eps=eps)
        )

    stacked = torch.stack(losses)

    if reduction == "mean":
        return stacked.mean()
    else:
        return stacked.sum()

def parabolic_hill_spec_layers(model, layer=2, **kwargs):
    weights = [p for n, p in model.named_parameters() if "weight" in n]

    if layer >= len(weights):
        raise ValueError(f"Layer index {layer} existiert nicht.")

    return parabolic_hill_single(weights[layer], **kwargs)



# def parabolic_hill_last_layers(model, n_layers=2, **kwargs):
#     weights = [p for n, p in model.named_parameters() if "weight" in n]
#     selected = weights[-n_layers:]

#     losses = [parabolic_hill_single(p, **kwargs) for p in selected]
#     return torch.stack(losses).mean()

    

# def parabolic_hill(model, k_ratio=0.1, reduction="sum", eps=1e-12): 
#     """Same as Standard Hill Regularizer, but scales alpha estimates with (1-alpha_hat)**2. Aim is to penalize values that are smaller and larger than 1 equally. 

#     Args:
#         model 
#         k_ratio (float, optional): Values that should be selected. Defaults to 0.1.
#         reduction (str, optional): Sum vs mean of alpha values. Defaults to "sum".
#         eps (float, optional): Small number for mumeric stability. Defaults to 1e-12.

#     Returns:
#         float: mean/sum of scaled alphas that will be added to loss function
#     """
#     losses = []

#     for name, param in model.named_parameters():
#         if "weight" not in name:
#             continue

#         x = param.view(-1).abs() + eps
#         x_sorted, _ = torch.sort(x, descending=True)
#         k = max(5, int(len(x_sorted) * k_ratio))
#         topk = x_sorted[:k]

#         logs = torch.log(topk + eps)
#         alpha_hat = 1.0 / (logs.mean() - logs[-1] + eps)

#         losses.append((1-alpha_hat)**2)

#         stacked = torch.stack(losses)
        
#     if reduction == "mean":
#         return stacked.mean()
#     else:
#         return stacked.sum()   


# -----------------------------------------------------------
# Paper Implementations
# -----------------------------------------------------------

def weighted_alpha_regularizer(model, eps=1e-12):
    """Implementation of alpha regularizer from Heavy-Tailed Regularization of Weight Matrices  in Deep Neural Networks (Xiao et al.)

    Args:
        model (_type_)
        eps (_type_, optional): Small number for numeric stability. Defaults to 1e-12.

    Returns:
        float: penalty that should be added to loss function
    """
    penalties = []

    for name, param in model.named_parameters():
        if "weight" not in name:
            continue
        
        W = param.view(param.size(0), -1)  
        S = W.T @ W  
        eigvals = torch.linalg.eigvalsh(S)  
        eigvals = torch.clamp(eigvals.real, min=eps)  

        x_sorted, _ = torch.sort(eigvals)
        n = len(x_sorted)
        k = max(5, n // 2)  
        topk = x_sorted[-k:]
        log_ratios = torch.log(topk / topk[0] + eps)
        alpha_l = 1 + k / (log_ratios.sum() + eps)

        lambda_max = eigvals.max()
        pl = alpha_l * torch.log(lambda_max + eps)
        penalties.append(pl)

    return torch.stack(penalties).sum()  

def decay_weighted_alpha_regularizer(model, epoch, decay_type="power", m=1, k_decay=1.0, t=0.0, eps=1e-12):
    """Implementation of alpha regularizer from Heavy-Tailed Regularization of Weight Matrices  in Deep Neural Networks (Xiao et al.)

    Args:
        model (_type_)
        eps (_type_, optional): Small number for numeric stability. Defaults to 1e-12.

    Returns:
        float: penalty that should be added to loss function
    """
    base_penalty = weighted_alpha_regularizer(model, eps=eps)
    x = epoch // m
    if decay_type == "power":
        decay = (x ** -k_decay) if (x ** -k_decay > t) else 0.0
    elif decay_type == "exp":
        decay = (torch.exp(-k_decay * x)).item() if (torch.exp(-k_decay * x) > t) else 0.0
    else:
        decay = 1.0
    return decay * base_penalty
  

def lower_threshold_weighted_alpha_regularizer(model, t=0.0, eps=1e-12):

    """Implementation of alpha regularizer from Heavy-Tailed Regularization of Weight Matrices  in Deep Neural Networks (Xiao et al.)

    Args:
        model (_type_)
        eps (_type_, optional): Small number for numeric stability. Defaults to 1e-12.

    Returns:
        float: penalty that should be added to loss function
    """

    base_penalty = weighted_alpha_regularizer(model, eps=eps)
    if base_penalty > t:
        return base_penalty
    else:
        return torch.tensor(0.0, device=next(model.parameters()).device)       
    

