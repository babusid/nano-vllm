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


def load_model(
    model: nn.Module,
    path: str,
    *,
    allow_unexpected: bool = False,
):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    safetensors_files = glob(os.path.join(path, "*.safetensors"))
    if safetensors_files:
        weight_iter = _iter_safetensors(path)
    else:
        weight_iter = _iter_pytorch_bin(path)

    # Non-parameter tensors (buffers like inv_freq) that appear in older
    # .bin checkpoints but are computed at init time — safe to skip.
    SKIP_SUFFIXES = (".inv_freq",)

    expected_param_names = {name for name, _ in model.named_parameters()}
    expected_buffer_names = {name for name, _ in model.named_buffers()}
    loaded_param_names = set()
    loaded_buffer_names = set()
    unexpected_keys = []
    checkpoint_dtypes = set()

    num_loaded = 0
    for weight_name, loaded_weight in weight_iter:
        if any(weight_name.endswith(s) for s in SKIP_SUFFIXES):
            continue
        checkpoint_dtypes.add(str(loaded_weight.dtype))
        for k in packed_modules_mapping:
            if k in weight_name:
                v, shard_id = packed_modules_mapping[k]
                param_name = weight_name.replace(k, v)
                param = model.get_parameter(param_name)
                weight_loader = getattr(param, "weight_loader")
                try:
                    weight_loader(param, loaded_weight, shard_id)
                except Exception as e:
                    raise RuntimeError(
                        "Failed loading packed checkpoint tensor "
                        f"{weight_name!r} with shape {tuple(loaded_weight.shape)} "
                        f"into model parameter {param_name!r} with shape "
                        f"{tuple(param.shape)}"
                    ) from e
                loaded_param_names.add(param_name)
                num_loaded += 1
                break
        else:
            try:
                param = model.get_parameter(weight_name)
            except AttributeError:
                # Not an nn.Parameter — try as a registered buffer. EAGLE-3
                # stores vocab-mapping tables (`d2t`, `t2d`) as buffers since
                # they're lookup tensors, not learned weights.
                try:
                    buf = model.get_buffer(weight_name)
                except AttributeError as e:
                    unexpected_keys.append(weight_name)
                    if allow_unexpected:
                        continue
                    raise RuntimeError(
                        f"Checkpoint key {weight_name!r} matches neither a "
                        f"parameter nor a buffer on the model"
                    ) from e
                try:
                    buf.copy_(loaded_weight.to(buf.dtype))
                except Exception as e:
                    raise RuntimeError(
                        "Failed loading checkpoint buffer "
                        f"{weight_name!r} with shape {tuple(loaded_weight.shape)} "
                        f"into model buffer with shape {tuple(buf.shape)}"
                    ) from e
                loaded_buffer_names.add(weight_name)
                num_loaded += 1
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            try:
                weight_loader(param, loaded_weight)
            except Exception as e:
                raise RuntimeError(
                    "Failed loading checkpoint tensor "
                    f"{weight_name!r} with shape {tuple(loaded_weight.shape)} "
                    f"into model parameter with shape {tuple(param.shape)}"
                ) from e
            loaded_param_names.add(weight_name)
            num_loaded += 1

    if num_loaded == 0:
        raise RuntimeError(
            f"No weights loaded from {path}. "
            "Expected *.safetensors or pytorch_model*.bin files."
        )

    missing_keys = sorted(
        (expected_param_names - loaded_param_names)
        | (expected_buffer_names - loaded_buffer_names)
    )
    return {
        "num_loaded": num_loaded,
        "missing_keys": missing_keys,
        "unexpected_keys": sorted(unexpected_keys),
        "checkpoint_dtypes": sorted(checkpoint_dtypes),
    }
