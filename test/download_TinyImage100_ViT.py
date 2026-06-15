import torch
from transformers import ViTModel, ViTImageProcessor
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ViT model and processor
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device).eval()
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Load TinyImageNet (adjust [:5%] for faster runs)
hf_ds = load_dataset("slegroux/tiny-imagenet-200-clean", split="train[:100%]")

# Wrapper for per-image transformation
class TinyImageNetViTDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),  # [3,224,224]
            "label": item["label"]
        }

# Wrap dataset and create DataLoader
wrapped_ds = TinyImageNetViTDataset(hf_ds, processor)
dl = DataLoader(wrapped_ds, batch_size=32, num_workers=4)

# Extract features
all_embs, all_labels = [], []
with torch.no_grad():
    for batch in dl:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"]
        outputs = model(pixel_values)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        all_embs.append(cls_embeddings.cpu().numpy())
        all_labels.append(np.array(labels))

# Save features
X = np.vstack(all_embs)
y = np.concatenate(all_labels)
np.save("/shared/Dataset/Clustering/tinyimagenet_vit_embs.npy", X)
np.save("/shared/Dataset/Clustering/tinyimagenet_vit_labels.npy", y)

print("✅ Saved embeddings:", X.shape)
print("✅ Saved labels:", y.shape)
