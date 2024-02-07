from genotypes import get_connection_dictionary
from modalities import sequential_mode, complete_mode, two_branch_mode, one_operation_mode
import torch
from torch.nn import BatchNorm2d
import types
from typing import Union, Text
from torch import nn
import numpy as np
from thop import profile  # https://github.com/Lyken17/pytorch-OpCounter
scale = 1e6

def sum_arr(arr):
    sum = 0.
    for i in range(len(arr)):
        sum += torch.sum(arr[i])
    return sum.item()

def get_layer_metric_array(net, metric, mode):
    metric_array = []

    for layer in net.modules():
        if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))

    return metric_array


def compute_naswot_score(net: nn.Module, inputs: torch.Tensor): # , targets: torch.Tensor, device: torch.device):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    with torch.no_grad():
        codes = []

        def hook(self: nn.Module, m_input: torch.Tensor, m_output: torch.Tensor):
            code = (m_output > 0).flatten(start_dim=1)
            codes.append(code)

        hooks = []
        for m in net.modules():
            if isinstance(m, torch.nn.ReLU) or isinstance(m,torch.nn.GELU):
                hooks.append(m.register_forward_hook(hook))

        _ = net(inputs)

        for h in hooks:
            h.remove()

        full_code = torch.cat(codes, dim=1)

        # Fast Hamming distance matrix computation
        del codes, _
        full_code_float = full_code.float()
        k = full_code_float @ full_code_float.t()
        del full_code_float
        not_full_code_float = torch.logical_not(full_code).float()
        k += not_full_code_float @ not_full_code_float.t()
        del not_full_code_float

        return torch.slogdet(k).logabsdet.item()

def _no_op(self, x):
    return x


# LogSynflow
def compute_synflow_per_weight(net, inputs, mode='param', remap: Union[Text, None] = 'log'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net = net.train()

    # Disable batch norm
    for layer in net.modules():
        if isinstance(layer, BatchNorm2d):
            # TODO: this could be done with forward hooks
            layer._old_forward = layer.forward
            layer.forward = types.MethodType(_no_op, layer)

    # Convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    # Convert to original values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    # Keep signs of all params
    signs = linearize(net)

    # Compute gradients with input of 1s
    net.zero_grad()
    net.double()
    input_dim = list(inputs[0, :].shape)
    inputs = torch.ones([1] + input_dim).double().to(device)
    output = net(inputs)
    if isinstance(output, tuple):
        output = output[1]
    torch.sum(output).backward()

    # Select the gradients that we want to use for search/prune
    def synflow(layer):
        if layer.weight.grad is not None:
            if remap:
                remap_fun = {
                    'log': lambda x: torch.log(x + 1),
                    # Other reparametrizations can be added here
                    # 'atan': torch.arctan,
                    # 'sqrt': torch.sqrt
                }
                # LogSynflow
                g = remap_fun[remap](layer.weight.grad)
            else:
                # Traditional synflow
                g = layer.weight.grad
            return torch.abs(layer.weight * g)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, synflow, mode)

    # Apply signs of all params
    nonlinearize(net, signs)

    # Enable batch norm again
    for layer in net.modules():
        if isinstance(layer, BatchNorm2d):
            layer.forward = layer._old_forward
            del layer._old_forward

    net.float()
    return sum_arr(grads_abs)


def count_params(model: nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params.item()


def get_macs_and_params(model: nn.Module, input_shape: list):
    model_device = next(model.parameters()).device
    input_ = torch.rand(input_shape, device=model_device)
    macs, params = profile(model, inputs=(input_, ), verbose=False)
    return macs / scale, params / scale

def skip(genotype):
    skip_connections = 0
    for block in genotype :
        connection_info =  get_connection_dictionary ( block[2] ) 
        if ( two_branch_mode(connection_info) ):
            continue

        if ( one_operation_mode(connection_info) ):
            continue

        if ( sequential_mode(connection_info) ):
            if connection_info["skip"] :
                skip_connections += 1
            else:
                continue

        if ( complete_mode(connection_info) ):
            if connection_info["skip"]:
                skip_connections += 2 #skip and input
            else : 
                skip_connections += 1 # input

    else :
        return skip_connections 