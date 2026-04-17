import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _iter_safetensors(path: str):
    """Yield (weight_name, tensor) pairs from safetensors files."""
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                yield weight_name, f.get_tensor(weight_name)


def _iter_pytorch_bin(path: str):
    """Yield (weight_name, tensor) pairs from pytorch .bin files."""
    for file in sorted(glob(os.path.join(path, "pytorch_model*.bin"))):
        state_dict = torch.load(file, map_location="cpu", weights_only=True)
        for weight_name, tensor in state_dict.items():
            yield weight_name, tensor
        del state_dict


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    safetensors_files = glob(os.path.join(path, "*.safetensors"))
    if safetensors_files:
        weight_iter = _iter_safetensors(path)
    else:
        weight_iter = _iter_pytorch_bin(path)

    # Non-parameter tensors (buffers like inv_freq) that appear in older
    # .bin checkpoints but are computed at init time — safe to skip.
    SKIP_SUFFIXES = (".inv_freq",)

    num_loaded = 0
    for weight_name, loaded_weight in weight_iter:
        if any(weight_name.endswith(s) for s in SKIP_SUFFIXES):
            continue
        for k in packed_modules_mapping:
            if k in weight_name:
                v, shard_id = packed_modules_mapping[k]
                param_name = weight_name.replace(k, v)
                param = model.get_parameter(param_name)
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                num_loaded += 1
                break
        else:
            param = model.get_parameter(weight_name)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            num_loaded += 1

    if num_loaded == 0:
        raise RuntimeError(
            f"No weights loaded from {path}. "
            "Expected *.safetensors or pytorch_model*.bin files."
        )
