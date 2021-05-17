import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device")