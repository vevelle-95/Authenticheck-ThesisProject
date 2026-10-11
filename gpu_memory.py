"""Set a conservative PyTorch allocation budget before CUDA model loading."""
import math


def configure_gpu_memory(gigabytes=None):
    if gigabytes is None:
        return
    if not math.isfinite(gigabytes) or gigabytes <= 0.3:
        raise ValueError("GPU budget must exceed 0.3 GB for CUDA overhead")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A GPU budget was requested, but CUDA is unavailable")
    total = torch.cuda.get_device_properties(0).total_memory
    # Leave 300 MB for context/library allocations outside PyTorch's allocator.
    limit = min((gigabytes - 0.3) * 1_000_000_000, total)
    torch.cuda.set_per_process_memory_fraction(limit / total, device=0)
    print(f"GPU target: {gigabytes:.2f} GB; PyTorch allocator cap: {limit / 1e9:.2f} GB. Other apps are outside this budget.", flush=True)
