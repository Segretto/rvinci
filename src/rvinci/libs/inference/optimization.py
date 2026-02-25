import torch
from rvinci.core.logging import get_logger
from rvinci.libs.gstreamer.utils import is_aarch64
from rvinci.libs.vision_data.constants import IMG_MEAN, IMG_STD
import numpy as np

log = get_logger(__name__)


def optimize_and_warmup(model: torch.nn.Module, config, device: str) -> torch.nn.Module:
    """
    Applies torch.compile (with TensorRT backend) and runs a warmup iteration
    only if running on Aarch64 (like Nvidia Jetson). Returns the original model otherwise.
    """
    if not is_aarch64():
        log.info("Not running on Aarch64. Skipping TensorRT compilation and warmup.")
        return model

    log.info("Running on Aarch64. Applying TensorRT compilation...")
    import importlib.util

    if importlib.util.find_spec("torch_tensorrt") is None:
        log.warning("torch_tensorrt not installed, skipping compilation.")
        return model

    opts = {
        "enabled_precisions": {torch.float16} if config.fp16 else {torch.float32},
        "torch_executed_ops": {"aten.native_group_norm.default"},
        "min_block_size": 10,
        "debug": False,
    }

    trt_model = torch.compile(model, backend="tensorrt", options=opts)

    # Warmup
    log.info("Warming up compiled model...")
    dtype = torch.float16 if config.fp16 else torch.float32

    # Use normalized input for warmup to avoid numerical instability
    example_input = torch.zeros(
        (1, 3, config.image_size, config.image_size), dtype=dtype, device=device
    )

    mean_np = np.array(IMG_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std_np = np.array(IMG_STD, dtype=np.float32).reshape(3, 1, 1)

    mean_tensor = torch.from_numpy(mean_np).to(device=device, dtype=dtype)
    std_tensor = torch.from_numpy(std_np).to(device=device, dtype=dtype)
    example_input = (example_input - mean_tensor) / std_tensor

    with torch.inference_mode():
        trt_model(example_input)

    log.info("Warmup complete.")
    return trt_model
