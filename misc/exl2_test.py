import sys
import torch
import os

#------------exllamav2 extension load test script----------------#
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version (Torch): {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")

try:
    import exllamav2
    print(f"ExLlamaV2 Version: {exllamav2.__version__}")
    print(f"ExLlamaV2 Location: {os.path.dirname(exllamav2.__file__)}")
except ImportError:
    print("ExLlamaV2 not installed.")

print("\n--- Attempting to load Extension ---")
try:
    # This is the specific import that is failing silently
    from exllamav2.ext import exllamav2_ext
    print("SUCCESS: Extension loaded directly!")
except Exception as e:
    print(f"FAILURE: Extension load failed.\nError: {e}")

#------------cuda and exllama test----------------#
import sys, torch, os
print("python:", sys.version.splitlines()[0])
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device props:", torch.cuda.get_device_properties(0))
try:
    import exllamav2
    print("exllamav2:", exllamav2.__version__, os.path.dirname(exllamav2.__file__))
    print("files in package:", os.listdir(os.path.dirname(exllamav2.__file__)))
except Exception as e:
    print("exllamav2 import error:", repr(e))