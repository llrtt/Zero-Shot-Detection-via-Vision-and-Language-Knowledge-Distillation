import os
import clip
import torch
from torchvision.datasets import CIFAR100
from vocDataset import vocData
import vocDataset

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
data = vocData("/home/llrt/文档/VOCdevkit/VOC2012")
# cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs
image, target = data[3637]
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in vocDataset.classes]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)
print(image_features.shape)
# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
print(image_features.shape)
text_features /= text_features.norm(dim=-1, keepdim=True)
print(text_features.shape)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
print(similarity.shape)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{vocDataset.classes[index]:>16s}: {100 * value.item():.2f}%")
