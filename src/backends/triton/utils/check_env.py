import sys

print("=== Python environment ===")
print("Python:", sys.version)

print("\n=== PyTorch check ===")
try:
    import torch

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA version used by PyTorch:", torch.version.cuda)
        print("GPU count:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0))

        x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        y = x * 2.0
        print("Torch CUDA test:", y.cpu().numpy())
    else:
        print("No CUDA GPU visible to PyTorch in this session.")

except ImportError:
    print("PyTorch is not installed.")

print("\n=== Triton check ===")
try:
    import triton

    print("Triton:", triton.__version__)
except ImportError:
    print("Triton is not installed.")
