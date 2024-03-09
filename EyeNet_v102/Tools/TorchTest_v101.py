import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms

## Simple tool to ensure CUDA is available

if torch.cuda.is_available():
        print("CUDA is available.")
        print("Number of CUDA devices:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print("CUDA device", i, ":", torch.cuda.get_device_name(i))
else:
    print("CUDA is not available.")
