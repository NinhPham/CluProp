from datasets import load_dataset
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import numpy as np

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Load Tiny ImageNet (use 1% for debugging)
hf_ds = load_dataset("slegroux/tiny-imagenet-200-clean", split="train")

# ImageNet transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Custom wrapper that avoids batching issues
class HFWrapper(Dataset):
    def __init__(self, hf_dataset, transform):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]  # a PIL Image
        label = item["label"]
        return {
            "pixel_values": self.transform(image),
            "label": label
        }

# Create dataset and dataloader
wrapped_ds = HFWrapper(hf_ds, transform)
dl = DataLoader(wrapped_ds, batch_size=64, shuffle=False, num_workers=4)

# Resnet50 has 2048 dimensional embeddings
# Load ResNet and strip classifier
# resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# resnet.fc = torch.nn.Identity()
# resnet.eval().cuda()

# Resnet10 has 512 dimensional embeddings
import timm
# Load ResNet-10 (from timm) and remove classification head
resnet = timm.create_model('resnet10t', pretrained=True)
resnet.reset_classifier(0)  # this replaces `fc = Identity()`
resnet.eval().cuda()

# Embed and collect
all_embs, all_labels = [], []
with torch.no_grad():
    for batch in dl:
        x = batch["pixel_values"].cuda()
        y = batch["label"]
        z = resnet(x).cpu().numpy()
        all_embs.append(z)
        all_labels.extend(y)

# Save to disk
X = np.vstack(all_embs)
y = np.array(all_labels)

np.save("tinyimagenet100_resnet10_embs.npy", X)
np.save("tinyimagenet100_resnet10_labels_98179_512.npy", y)

print("Done. Saved:", X.shape, y.shape)
