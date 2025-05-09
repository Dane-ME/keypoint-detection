import torch
print("true" if torch.cuda.is_available() else "false")  # Should print True if CUDA is available
print(torch.version.cuda)  # Should show CUDA version