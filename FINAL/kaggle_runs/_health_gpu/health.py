import torch, sys
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
try:
    x = torch.randn(64, 64, device="cuda")
    y = (x @ x).sum().item()      # real compute kernel launch
    print(f"CUDA-COMPUTE-OK matmul_sum={y:.2f}")
except Exception as e:
    print(f"CUDA-COMPUTE-FAIL: {type(e).__name__}: {e}")
    sys.exit(1)
